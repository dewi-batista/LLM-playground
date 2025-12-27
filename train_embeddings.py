# Embedding training (word2vec-ish SGNS)
# TODO: hyperparameter tuning; lr, d, k, subsample_t, min_count

from datetime import datetime
from itertools import pairwise
from pathlib import Path
from tqdm import tqdm

import json
import numpy as np
import pickle
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

HERE = Path(__file__).resolve().parent

# NOTE: I learned recently of an "mps" device which some Apple MacBooks have.
# Perhaps worth adding in future (my M1 MacBook Air blows).
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

VOCAB_PATH = None  # set to a specific file or None for latest BPE vocab

# for vocab
def latest_bpe_vocab(data_dir: Path) -> Path | None:
    candidates = [
        p
        for p in data_dir.glob("vocabulary_bpe_*.json")
        if p.is_file() and len(p.stem.split("_")[-1]) == 14 and p.stem.split("_")[-1].isdigit()
    ]
    return max(candidates, key=lambda p: p.stem.split("_")[-1]) if candidates else None

if VOCAB_PATH is None:
    VOCAB_PATH = latest_bpe_vocab(HERE / "data") or (HERE / "./data/vocabulary.json")

with open(VOCAB_PATH) as f:
    vocab = json.load(f)

# for config
with open(HERE / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# model hyperparams
batch_size = 8_192
d = int(config["model"]["d_model"])
epochs = 5
k = 10
lr = 1e-3
min_count = 5
steps_per_epoch = 100_000
subsample_t = 1e-5
total_steps = epochs * steps_per_epoch
window = 5

# other hyperparams
log_every = 1_000

def iter_pre_tokens(sequence: str):
    sequence = sequence.strip()
    first = True
    for match in re.finditer(r"\S+", sequence):
        token = match.group(0)
        if first:
            yield token
            first = False
        else:
            yield " " + token

is_bpe_vocab = all(k.isdigit() for k in vocab.keys())
if is_bpe_vocab:
    vocab_timestamp = VOCAB_PATH.stem.split("_")[-1]
    encodings_path = HERE / f"./config/encodings_{vocab_timestamp}.pkl"
    with open(encodings_path, "rb") as f:
        encodings = pickle.load(f)

    V_full = len(vocab)
    keep_token_ids = [i for i in range(V_full) if int(vocab[str(i)]["count"]) >= min_count]
    index_to_token = [vocab[str(i)]["string"] for i in keep_token_ids]
    V = len(index_to_token)

    # for probabilities pertaining to negative indices
    neg_probs = np.zeros(V, dtype=np.float64)
    counts = np.zeros(V, dtype=np.int64)
    for idx, token_id in enumerate(keep_token_ids):
        info = vocab[str(token_id)]
        neg_probs[idx] = float(info["neg_prob"])
        counts[idx] = int(info["count"])
    neg_probs /= neg_probs.sum()
    neg_probs_t = torch.as_tensor(neg_probs, dtype=torch.float, device=device)

    # tokenise text8.txt into token IDs (cached)
    token_ids_path = HERE / f"./data/text8_token_ids_{vocab_timestamp}.npy"
    if token_ids_path.exists():
        token_ids = np.load(token_ids_path, mmap_mode="r")
    else:
        merges = {tuple(pair): new_token for pair, new_token in encodings}
        ranks = {tuple(pair): i for i, (pair, _) in enumerate(encodings)}
        cache = {}

        def bpe_encode(token: str):
            if token in cache:
                return cache[token]
            ids = list(token.encode("utf-8"))
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
            cache[token] = ids
            return ids

        with open(HERE / "./data/text8.txt") as f:
            corpus = f.read()

        total_token_ids = sum(int(info["count"]) for info in vocab.values())
        token_ids = np.empty(total_token_ids, dtype=np.uint16 if V_full < 65536 else np.int32)
        pos = 0
        for token in tqdm(iter_pre_tokens(corpus), desc="tokenising text8", unit="token"):
            ids = bpe_encode(token)
            token_ids[pos : pos + len(ids)] = ids
            pos += len(ids)
        token_ids = token_ids[:pos]
        np.save(token_ids_path, token_ids)

    # map token IDs to embedding indices (and drop pruned tokens)
    token_id_to_index = np.full(V_full, -1, dtype=np.int32)
    for i, token_id in enumerate(keep_token_ids):
        token_id_to_index[token_id] = i
    indeces_corpus_to_token = token_id_to_index[token_ids]
    indeces_corpus_to_token = indeces_corpus_to_token[indeces_corpus_to_token >= 0]

else:
    # for text
    with open(HERE / "./data/text8.txt") as f:
        corpus_text = f.read().split()

    # prune vocab
    vocab_keep = [(int(info["index"]), word) for word, info in vocab.items() if int(info["count"]) >= min_count]
    vocab_keep.sort(key=lambda x: x[0])
    index_to_token = [word for _, word in vocab_keep]
    token_to_index = {word: i for i, word in enumerate(index_to_token)}
    V = len(index_to_token)

# config for storing model params
models_dir = HERE / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# Set RESUME_FROM to None for a new model, "latest" to resume form most recent
# checkpoint and to a given checkpoint path to resume training from said state.
RESUME_FROM = None

def latest_checkpoint(models_dir):
    candidates = [p for p in models_dir.glob("*.ckpt")]
    return max(candidates, key=lambda p: p.stem)

if RESUME_FROM == "latest":
    checkpoint_path = latest_checkpoint(models_dir)
elif RESUME_FROM is not None:
    checkpoint_path = Path(RESUME_FROM)
else:
    run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    checkpoint_path = models_dir / f"{run_timestamp}.ckpt"

if not is_bpe_vocab:
    # for probabilities pertaining to negative indices
    neg_probs = np.zeros(V, dtype=np.float64)
    counts = np.zeros(V, dtype=np.int64)
    for idx, word in enumerate(index_to_token):
        info = vocab[word]
        neg_probs[idx] = float(info["neg_prob"])
        counts[idx] = int(info["count"])
    neg_probs /= neg_probs.sum()
    neg_probs_t = torch.as_tensor(neg_probs, dtype=torch.float, device=device)

# subsampling frequent words
freqs = counts.astype(np.float64) / float(counts.sum())
subsample_keep_probs = (np.sqrt(freqs / subsample_t) + 1.0) * (subsample_t / freqs)
subsample_keep_probs = np.minimum(subsample_keep_probs, 1.0)

def neg_sampling_loss(u, v, U, context_idx):
    pos_scores = torch.sum(u * v, dim=1)  # (B,)
    loss = -F.logsigmoid(pos_scores)  # (B,)

    neg_indeces = torch.multinomial(neg_probs_t, num_samples=u.shape[0] * k, replacement=True)
    neg_indeces = neg_indeces.view(u.shape[0], k)  # (B, k)
    context_idx = context_idx.unsqueeze(1)  # (B, 1)
    for _ in range(2):
        mask = neg_indeces.eq(context_idx)
        replacement = torch.multinomial(neg_probs_t, num_samples=u.shape[0] * k, replacement=True).view(
            u.shape[0], k
        )
        neg_indeces = torch.where(mask, replacement, neg_indeces)
    mask = neg_indeces.eq(context_idx)
    neg_indeces = torch.where(mask, (neg_indeces + 1) % V, neg_indeces)
    neg_vecs = U(neg_indeces)  # (B, k, d)
    neg_scores = torch.sum(neg_vecs * u.unsqueeze(1), dim=2)  # (B, k)
    loss -= torch.sum(F.logsigmoid(-neg_scores), dim=1)  # (B,)

    return torch.mean(loss)

if not is_bpe_vocab:
    # to go straight from index in text8.txt to token index according to vocabulary
    indeces_corpus_to_token = np.array(
        [token_to_index[word] for word in corpus_text if word in token_to_index], dtype=np.int32
    )
subsample_seed = int(checkpoint_path.stem) % (2**32 - 1) if checkpoint_path.stem.isdigit() else 0
subsample_rng = np.random.RandomState(subsample_seed)
subsample_mask = subsample_rng.random_sample(size=len(indeces_corpus_to_token)) < subsample_keep_probs[indeces_corpus_to_token]
indeces_corpus_to_token = indeces_corpus_to_token[subsample_mask]
corpus_len = len(indeces_corpus_to_token)

E = nn.Embedding(V, d).to(device)
U = nn.Embedding(V, d).to(device)
optimizer = optim.Adam(list(E.parameters()) + list(U.parameters()), lr=lr)
start_epoch = 0
if RESUME_FROM is not None:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    E.load_state_dict(ckpt["E_state_dict"])
    U.load_state_dict(ckpt["U_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = int(ckpt["epoch"])

    random.setstate(ckpt["rng_state_py"])
    np.random.set_state(ckpt["rng_state_np"])
    torch.set_rng_state(ckpt["rng_state_torch"].cpu())
    if torch.cuda.is_available() and ckpt["rng_state_cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng_state_cuda"]])
    tqdm.write(f"resuming from: {checkpoint_path} (epoch={start_epoch})")

# NOTE: Epochs aren't actually needed here as there's no fixed dataset to
# iterate over. What's implemented is entirely equivalent to instead performing
# epochs * steps_per_epoch optimisation steps. Despite this, epochs are kept
# for the convenience checkpointing loss, parameters, etc.

for epoch in range(start_epoch, epochs):
    total_loss = 0.0
    log_loss = 0.0
    log_steps = 0
    pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch + 1}/{epochs}", unit="step")
    for step in pbar:
        global_step = epoch * steps_per_epoch + step
        current_lr = lr * (1.0 - (global_step / total_steps))
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        
        # sample the center word and context word
        idx_centers = np.random.randint(window, corpus_len - window, size=batch_size)
        window_sizes = np.random.randint(1, window + 1, size=batch_size)
        magnitudes = (np.random.random(size=batch_size) * window_sizes).astype(np.int32) + 1
        signs = np.where(np.random.random(size=batch_size) < 0.5, -1, 1).astype(np.int32)
        offsets = signs * magnitudes
        idx_contexts = idx_centers + offsets
        center_idx = torch.as_tensor(indeces_corpus_to_token[idx_centers], dtype=torch.long, device=device)
        context_idx = torch.as_tensor(indeces_corpus_to_token[idx_contexts], dtype=torch.long, device=device)

        optimizer.zero_grad()

        emb_center = E(center_idx)
        emb_contxt = U(context_idx)

        loss = neg_sampling_loss(emb_center, emb_contxt, U, context_idx)
        loss.backward()
        optimizer.step()
        loss_val = float(loss.detach())
        total_loss += loss_val
        log_loss += loss_val
        log_steps += 1

        if log_every > 0 and (step + 1) % log_every == 0:
            pbar.set_postfix(recent_loss=f"{log_loss / log_steps:.4f}")
            log_loss = 0.0
            log_steps = 0

    avg_loss = total_loss / steps_per_epoch
    tqdm.write(f"epoch {epoch + 1}/{epochs} avg_loss={avg_loss:.4f}")
    
    # NOTE: What's saved is for identical continuation. This assumes that saved
    # checkpoints correspond to fully completed epochs.
    torch.save(
        {
            "E_state_dict": E.state_dict(),
            "U_state_dict": U.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": V,
            "min_count": min_count,
            "index_to_token": index_to_token,
            "d_model": d,
            "is_bpe_vocab": bool(is_bpe_vocab),
            "bpe_vocab_path": str(VOCAB_PATH) if is_bpe_vocab else None,
            "bpe_encodings_path": str(encodings_path) if is_bpe_vocab else None,
            "epoch": epoch + 1,
            "rng_state_py": random.getstate(),
            "rng_state_np": np.random.get_state(),
            "rng_state_torch": torch.get_rng_state(),
            "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        checkpoint_path,
    )
    tqdm.write(f"saved: {checkpoint_path}")
