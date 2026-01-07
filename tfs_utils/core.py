from itertools import pairwise

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

PUNCT_SUFFIXES = [",", ".", ":", ";", "!", "?"]
PUNCT_PREFIXES = ["-"]


def iter_pre_tokens(text: str):
    text = text.strip()
    first = True
    for match in re.finditer(r"\S+", text):
        tok = match.group(0)
        if first:
            yield tok
            first = False
        else:
            yield " " + tok


def make_bpe_encoder(encodings):
    merges = {tuple(pair): new_token for pair, new_token in encodings}
    ranks = {tuple(pair): i for i, (pair, _) in enumerate(encodings)}
    cache = {}

    def encode(text: str):
        if text in cache:
            return cache[text]
        ids = list(text.encode("utf-8"))
        while True:
            best_pair = None
            best_rank = None
            for p in pairwise(ids):
                r = ranks.get(p)
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_pair = p
            if best_pair is None:
                break
            new_token = merges[best_pair]
            merged = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        cache[text] = ids
        return ids

    return encode


def positional_encoding(seq_len: int, d_model: int, device: torch.device):
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    i = torch.arange(d_model, device=device, dtype=torch.float32)  # (d,)
    div = torch.pow(10_000.0, (2 * (i // 2)) / d_model)
    angles = pos / div

    pe = torch.zeros_like(angles)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_QKV = nn.Linear(d_model, 3 * d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, X):
        B, T, _ = X.shape
        H = self.ln1(X)
        Q, K, V = self.W_QKV(H).chunk(3, dim=-1)
        Q = Q.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)

        O = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        O = O.transpose(1, 2).reshape(B, T, self.num_heads * self.d_head)
        X = X + self.dropout_attn(self.W_O(O))

        H = self.ln2(X)
        X = X + self.dropout_ffn(self.W_2(self.act(self.W_1(H))))
        return X


def token_to_cli(token: str) -> str:
    if token.startswith(" "):
        n = len(token) - len(token.lstrip(" "))
        return ("_" * n) + token[n:]
    return token


def encode_pre_tokens_to_indices(pre_tokens, bpe_encode, token_id_to_index):
    token_ids = []
    for tok in pre_tokens:
        token_ids.extend(bpe_encode(tok))
    if not token_ids:
        return []
    token_ids = torch.as_tensor(token_ids, dtype=torch.long)
    idx = token_id_to_index[token_ids]
    idx = idx[idx >= 0]
    return idx.tolist()


def target_variants(token: str) -> list[str]:
    base = token.rstrip("".join(PUNCT_SUFFIXES))
    variants = [token]
    if base and base != token:
        variants.append(base)
    for p in PUNCT_SUFFIXES:
        if base:
            variants.append(base + p)

    stripped = base.lstrip(" ")
    leading = base[: len(base) - len(stripped)]
    for p in PUNCT_PREFIXES:
        if stripped:
            variants.append(p + stripped)
            if leading:
                variants.append(leading + p + stripped)
            for s in PUNCT_SUFFIXES:
                variants.append(p + stripped + s)
                if leading:
                    variants.append(leading + p + stripped + s)

    seen = set()
    out = []
    for v in variants:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def build_token_id_to_index(vocab: dict, index_to_token: list[str]):
    token_str_to_index = {t: i for i, t in enumerate(index_to_token)}
    token_id_to_index = torch.full((len(vocab),), -1, dtype=torch.long)
    for token_id, info in vocab.items():
        idx = token_str_to_index.get(info["string"])
        if idx is not None:
            token_id_to_index[int(token_id)] = idx
    return token_id_to_index, token_str_to_index


def sample_next_token(
    logits: torch.Tensor,
    prev_tokens: list[int],
    *,
    sample: bool,
    temperature: float,
    repetition_penalty: float,
    no_repeat_ngram: int,
) -> int:
    logits = logits.clone()

    if repetition_penalty and repetition_penalty != 1.0 and prev_tokens:
        t = torch.as_tensor(list(set(prev_tokens)), device=logits.device)
        v = logits[t]
        logits[t] = torch.where(v > 0, v / repetition_penalty, v * repetition_penalty)

    n = int(no_repeat_ngram)
    if n > 0 and len(prev_tokens) >= (n - 1):
        seen = {}
        for i in range(len(prev_tokens) - n + 1):
            prefix = tuple(prev_tokens[i : i + n - 1])
            nxt = prev_tokens[i + n - 1]
            if prefix in seen:
                seen[prefix].add(nxt)
            else:
                seen[prefix] = {nxt}
        banned = seen.get(tuple(prev_tokens[-(n - 1) :]), set())
        if banned:
            logits[list(banned)] = -float("inf")

    if temperature and temperature != 1.0:
        logits = logits / float(temperature)

    if not sample:
        return int(torch.argmax(logits).item())

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def next_token_logits(prompt_indeces, E, model, final_lay_norm, U, pe):
    x = torch.as_tensor(prompt_indeces, dtype=torch.long, device=E.weight.device).unsqueeze(0)  # (1, T)
    X = E(x) + pe[: x.shape[1]]
    return U(final_lay_norm(model(X)))[0, -1]  # (V,)
