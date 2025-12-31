# TODO: read about warmup + cosine LR schedules
# TODO: read about gradient accumulation (multiple forward/backward passes before stepping)
# TODO: read about gradient clipping (stabilises training)
# TODO: read about warmup + cosine LR schedules
# TODO: read about AdamW + weight decay (and why you often exclude biases/norms)
# TODO: read about why head dim is often ~64 (so n_heads ≈ d_model/64)
# TODO: read about splitting params into decay/no-decay groups (biases + norms usually no decay)
# TODO: read about AdamW defaults (betas/eps) used for transformers

from cache_tokenisation import load_or_create_token_ids
from pathlib import Path
from tqdm import tqdm

import json
import math
import numpy as np
import pickle
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# command line input
if len(sys.argv) < 3 or sys.argv[1] in {"-h", "--help"}:
    print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp> [model_number]")
    raise SystemExit(1)
language = sys.argv[1]
timestamp = sys.argv[2]
model_number = int(sys.argv[3]) if (len(sys.argv) > 3) else None
resume = len(sys.argv) > 3

# NOTE: I learned recently of an "mps" device which some Apple MacBooks have.
# Perhaps worth consdering in the future (my current M1 MacBook Air blows).
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# directory shenanigans
HERE = Path(__file__).resolve().parent

run_dir = HERE / "models" / language / timestamp
run_dir.mkdir(parents=True, exist_ok=True)

corpus_path = HERE / "data" / f"{language}.txt"
encodings_path = run_dir / f"{language}_{timestamp}.pkl"
token_ids_path = run_dir / f"{language}_{timestamp}.npy"
vocab_path = run_dir / f"{language}_{timestamp}.json"

with open(HERE / "./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# NOTE: The following lines are pre-training sanity checks.
tqdm.write(f"device: {device} (cuda={torch.cuda.is_available()}, cuda_devices={torch.cuda.device_count()})")
if torch.cuda.is_available():
    tqdm.write(f"cuda[0]: {torch.cuda.get_device_name(0)}")
tqdm.write(f"run_dir: {run_dir}")
tqdm.write(f"corpus_path: {corpus_path} (exists={corpus_path.exists()})")
tqdm.write(f"encodings_path: {encodings_path} (exists={encodings_path.exists()})")
tqdm.write(f"vocab_path: {vocab_path} (exists={vocab_path.exists()})")
tqdm.write(f"token_ids_path: {token_ids_path} (exists={token_ids_path.exists()})")
if token_ids_path.exists():
    tqdm.write(f"token_ids_path size: {token_ids_path.stat().st_size / (1024**2):.1f} MB")

with open(encodings_path, "rb") as f:
    encodings = pickle.load(f)

with open(vocab_path) as f:
    vocab = json.load(f)
vocab_size = len(vocab)
tqdm.write(f"loaded encodings: {len(encodings)} merges")
tqdm.write(f"loaded vocab: {vocab_size} tokens")

# config hyperparams
batch_size = int(config["transformer"]["batch_size"])
d_model = int(config["transformer"]["d_model"])
dropout = 0.1
eval_batches = 50
eval_every = 200
grad_accum_steps = 1
grad_clip = 1.0
log_every = 20
lr = float(config["transformer"]["lr"])
min_count = 5
num_blocks = int(config["transformer"]["num_blocks"])
seq_len = 512
train_tokens = 1e9
val_frac = 0.01
warmup_frac = 0.02
weight_decay = 0.1

# non-config hyperparams
num_heads = d_model // 64 # so d_head = d_model / num_heads = 64
d_ff = 4 * d_model

# prune vocab of sufficiently-infrequent tokens
keep_token_ids = [i for i in range(vocab_size) if int(vocab[str(i)]["count"]) >= min_count]
index_to_token = [vocab[str(i)]["string"] for i in keep_token_ids]
V = len(index_to_token) # TODO: understand discrepancy between vocab_size and V
tqdm.write(f"pruned vocab: min_count={min_count} -> V={V}")

t0 = time.perf_counter()
token_ids = load_or_create_token_ids(
    language,
    timestamp,
    corpus_path=corpus_path,
    token_ids_path=token_ids_path,
    encodings=encodings,
    vocab=vocab,
)
tqdm.write(f"loaded token_ids: len={len(token_ids):,} dtype={token_ids.dtype} ({token_ids.nbytes / (1024**2):.1f} MB) in {time.perf_counter() - t0:.1f}s")

# map token IDs to embedding indices (and drop pruned tokens)
token_id_to_index = np.full(vocab_size, -1, dtype=np.int32)
for i, token_id in enumerate(keep_token_ids):
    token_id_to_index[token_id] = i
tqdm.write("building indeces_corpus_to_token (this can be slow on large corpora)...")
t0 = time.perf_counter()
indeces_corpus_to_token = token_id_to_index[token_ids]
indeces_corpus_to_token = indeces_corpus_to_token[indeces_corpus_to_token >= 0]
tqdm.write(f"built indeces_corpus_to_token: len={len(indeces_corpus_to_token):,} in {time.perf_counter() - t0:.1f}s")

# new ckpt if model number not passed as argument
if model_number is None:
    model_number = len(list(run_dir.glob(f"{language}_transformer_{timestamp}_*.ckpt"))) + 1
    checkpoint_path = run_dir / f"{language}_transformer_{timestamp}_{model_number}.ckpt"
    resume = False
else:
    checkpoint_path = run_dir / f"{language}_transformer_{timestamp}_{model_number}.ckpt"
    resume = True
tqdm.write(f"checkpoint_path: {checkpoint_path} (resume={resume})")
corpus_len = len(indeces_corpus_to_token)

tokens_per_step = batch_size * seq_len * grad_accum_steps
total_steps = int(math.ceil(train_tokens / tokens_per_step))
warmup_steps = max(1, int(total_steps * warmup_frac))
tqdm.write(f"train_tokens={train_tokens:,} tokens_per_step={tokens_per_step:,} total_steps={total_steps:,} warmup_steps={warmup_steps:,}")

# split corpus into train/val (contiguous split, LM-style)
val_start = int(corpus_len * (1.0 - val_frac))
train_token_ids = indeces_corpus_to_token[:val_start]
val_token_ids = indeces_corpus_to_token[val_start:]
tqdm.write(f"train/val split: train_len={len(train_token_ids):,} val_len={len(val_token_ids):,} (val_frac={val_frac})")

def positional_encoding(seq_len, d_model, device):
    positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1) # (n, 1)

    i = torch.arange(d_model, device=device, dtype=torch.float32)  # (d,)
    div_term = torch.pow(10_000.0, (2 * (i // 2)) / d_model)

    angles = positions / div_term  # (n, d)

    pe = torch.zeros_like(angles)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])

    return pe

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout):
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
        Q, K, V = self.W_QKV(X).chunk(3, dim=-1)
        Q = Q.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)

        M = torch.triu(X.new_full((T, T), float("-inf")), diagonal=1)
        A = torch.softmax((Q @ K.transpose(-2, -1)) / (self.d_head**0.5) + M, dim=-1)
        O = A @ V
        O = O.transpose(1, 2).reshape(B, T, self.num_heads * self.d_head)
        O = self.dropout_attn(self.W_O(O))
        H_1 = self.ln1(X + O)

        H_2 = self.dropout_ffn(self.W_2(self.act(self.W_1(H_1))))
        H_3 = self.ln2(H_1 + H_2)
        return H_3

dropout_embed = nn.Dropout(dropout).to(device)
E = nn.Embedding(V, d_model).to(device)
final_lay_norm = nn.LayerNorm(d_model).to(device)
model = nn.Sequential(*[TransformerBlock(d_model, d_ff, num_heads, dropout) for _ in range(num_blocks)]).to(device)
U = nn.Linear(d_model, V, bias=False).to(device)

# TODO: check that this is what it's intended to be (weight tying)
U.weight = E.weight

# with fixed seq_len, positional encoding only needs to be computed once
pe = positional_encoding(seq_len, d_model, device=device)

# NOTE: Does not include U.parameters() due to weight tying
params = list(E.parameters()) + list(model.parameters()) + list(final_lay_norm.parameters())
decay_params = [p for p in params if p.ndim >= 2]
no_decay_params = [p for p in params if p.ndim < 2]
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=lr,
    betas=(0.9, 0.95),
    eps=1e-8,
)

start_step = 0
best_val_ppl = float("inf")
if resume:
    assert checkpoint_path.exists()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    E.load_state_dict(ckpt["E_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"])
    final_lay_norm.load_state_dict(ckpt["final_lay_norm_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = int(ckpt.get("global_step", 0))
    best_val_ppl = float(ckpt.get("best_val_ppl", float("inf")))

    random.setstate(ckpt["rng_state_py"])
    np.random.set_state(ckpt["rng_state_np"])
    torch.set_rng_state(ckpt["rng_state_torch"].cpu())
    if torch.cuda.is_available() and ckpt["rng_state_cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng_state_cuda"]])
    tqdm.write(f"resuming from: {checkpoint_path} (step={start_step}, best_val_ppl={best_val_ppl:.2f})")

offsets = np.arange(seq_len, dtype=np.int64)

# NOTE: val_perplexity() offers an estimate of validation perplefxity due to
# its use of random batches, not being over the full corpus.
def val_perplexity():
    model.eval()
    E.eval()
    final_lay_norm.eval()
    U.eval()
    dropout_embed.eval()
    with torch.no_grad():
        total_loss = 0.0
        for _ in range(eval_batches):
            window_start_idx = np.random.randint(0, len(val_token_ids) - seq_len - 1, size=batch_size)
            context_window = val_token_ids[window_start_idx[:, None] + offsets[None, :]]
            targets = val_token_ids[window_start_idx[:, None] + offsets[None, :] + 1]

            context_window = torch.as_tensor(context_window, dtype=torch.long, device=device)
            targets = torch.as_tensor(targets, dtype=torch.long, device=device)

            X = dropout_embed(E(context_window) + pe)
            logits = U(final_lay_norm(model(X)))
            loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
            total_loss += float(loss)

    model.train()
    E.train()
    final_lay_norm.train()
    U.train()
    dropout_embed.train()

    mean_loss = total_loss / eval_batches
    return math.exp(mean_loss)

pbar = tqdm(range(start_step, total_steps), desc="train", unit="step", total=total_steps, initial=start_step)
log_loss = 0.0
log_time = 0.0
log_steps = 0
for step in pbar:
    step_t0 = time.perf_counter()
    if step < warmup_steps:
        current_lr = lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        current_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = current_lr # update learning rate of all params
    
    optimizer.zero_grad()

    step_loss = 0.0
    for _ in range(grad_accum_steps):
        # sample batch of token sequences (t_i, ..., t_{i + seq_len - 1})
        window_start_idx = np.random.randint(0, len(train_token_ids) - seq_len - 1, size=batch_size)
        context_window = train_token_ids[window_start_idx[:, None] + offsets[None, :]]
        targets = train_token_ids[window_start_idx[:, None] + offsets[None, :] + 1]

        context_window = torch.as_tensor(context_window, dtype=torch.long, device=device)
        targets = torch.as_tensor(targets, dtype=torch.long, device=device)

        X = dropout_embed(E(context_window) + pe)
        logits = U(final_lay_norm(model(X)))
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        (loss / grad_accum_steps).backward()
        step_loss += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(params, grad_clip)
    optimizer.step()

    log_time += time.perf_counter() - step_t0
    log_loss += step_loss / grad_accum_steps
    log_steps += 1

    if (step + 1) % log_every == 0:
        step_s = log_time / log_steps
        tok_s = tokens_per_step / step_s
        pbar.set_postfix(
            recent_loss=f"{log_loss / log_steps:.4f}",
            lr=f"{current_lr:.2e}",
            step_ms=f"{step_s * 1000:.1f}",
            tok_s=f"{tok_s / 1e6:.2f}M",
        )
        log_time = 0.0
        log_loss = 0.0
        log_steps = 0

    # checkpoint via validation perplexity
    if (step + 1) % eval_every == 0 or (step + 1) == total_steps:
        val_ppl = val_perplexity()
        pbar.set_postfix(val_ppl=f"{val_ppl:.2f}")
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(
                {
                    "E_state_dict": E.state_dict(),
                    "model_state_dict": model.state_dict(),
                    "final_lay_norm_state_dict": final_lay_norm.state_dict(),
                    "U_state_dict": U.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab_size": V,
                    "min_count": min_count,
                    "index_to_token": index_to_token,
                    "d_model": d_model,
                    "num_heads": num_heads,
                    "num_blocks": num_blocks,
                    "d_ff": d_ff,
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "dropout": dropout,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "grad_clip": grad_clip,
                    "warmup_frac": warmup_frac,
                    "warmup_steps": warmup_steps,
                    "total_steps": total_steps,
                    "grad_accum_steps": grad_accum_steps,
                    "train_tokens": train_tokens,
                    "tokens_per_step": tokens_per_step,
                    "val_frac": val_frac,
                    "eval_every": eval_every,
                    "eval_batches": eval_batches,
                    "best_val_ppl": best_val_ppl,
                    "val_ppl": val_ppl,
                    "bpe_vocab_path": str(vocab_path.relative_to(HERE)),
                    "bpe_encodings_path": str(encodings_path.relative_to(HERE)),
                    "global_step": step + 1,
                    "rng_state_py": random.getstate(),
                    "rng_state_np": np.random.get_state(),
                    "rng_state_torch": torch.get_rng_state(),
                    "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                checkpoint_path,
            )

def token_to_cli(token: str) -> str:
    if token.startswith(" "):
        n = len(token) - len(token.lstrip(" "))
        return ("_" * n) + token[n:]
    return token

model.eval()
E.eval()
final_lay_norm.eval()
U.eval()
dropout_embed.eval()
with torch.no_grad():
    for _ in range(5):
        start = np.random.randint(0, corpus_len - seq_len - 1)
        context = indeces_corpus_to_token[start : start + seq_len]
        true_next = int(indeces_corpus_to_token[start + seq_len])

        x = torch.as_tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        X = dropout_embed(E(x) + pe)
        logits = U(final_lay_norm(model(X)))[0, -1]  # (V,)
        probs = torch.softmax(logits, dim=-1)
        values, indices = torch.topk(probs, k=10)

        context_text = "".join(index_to_token[int(t)] for t in context[-10:])
        top5 = [(token_to_cli(index_to_token[int(j)]), float(v)) for v, j in zip(values, indices)]
        print("\n---")
        print(context_text)
        print("Target:", token_to_cli(index_to_token[true_next]))
        print("Top 5:", top5)
# keep this here do not indent!!!
# tqdm.write(f"saved: {checkpoint_path}")
