# TODO: hyperparameter tuning; lr, d, k, subsample_t, min_count
from itertools import pairwise
from pathlib import Path
from tqdm import tqdm

import json
import numpy as np
import pickle
import random
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

# command line input
if len(sys.argv) < 3 or sys.argv[1] in {"-h", "--help"}:
    print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp>")
    raise SystemExit(1)
language = sys.argv[1]
timestamp = sys.argv[2]

# NOTE: I learned recently of an "mps" device which some Apple MacBooks have.
# Perhaps worth adding in future (my M1 MacBook Air blows).
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

with open(encodings_path, "rb") as f:
    encodings = pickle.load(f)

with open(vocab_path) as f:
    vocab = json.load(f)
vocab_size = len(vocab)

# hyperparams
batch_size = 8_192
context_length = 128
d = int(config["model"]["d_model"])
epochs = 10
k = 10
lr = 1e-3
min_count = 5
steps_per_epoch = 100_000
subsample_t = 1e-5
window = 5

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

# prune vocab of sufficiently-infrequent tokens
keep_token_ids = [i for i in range(vocab_size) if int(vocab[str(i)]["count"]) >= min_count]
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

# cache the tokenisation of the corpus into token IDs (if not done already)
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

    with open(corpus_path) as f:
        corpus = f.read()

    total_token_ids = sum(int(info["count"]) for info in vocab.values())
    token_ids = np.empty(total_token_ids, dtype=np.uint16 if vocab_size < 65_536 else np.int32)
    pos = 0
    for token in tqdm(iter_pre_tokens(corpus), desc=f"tokenising {language}", unit="token"):
        ids = bpe_encode(token)
        token_ids[pos : pos + len(ids)] = ids
        pos += len(ids)
    token_ids = token_ids[:pos]
    np.save(token_ids_path, token_ids)

# map token IDs to embedding indices (and drop pruned tokens)
token_id_to_index = np.full(vocab_size, -1, dtype=np.int32)
for i, token_id in enumerate(keep_token_ids):
    token_id_to_index[token_id] = i
indeces_corpus_to_token = token_id_to_index[token_ids]
indeces_corpus_to_token = indeces_corpus_to_token[indeces_corpus_to_token >= 0]

# new ckpt
model_number = len(list(run_dir.glob(f"{language}_{timestamp}_*.ckpt"))) + 1
checkpoint_path = run_dir / f"{language}_{timestamp}_{model_number}.ckpt"
corpus_len = len(indeces_corpus_to_token)

E = nn.Embedding(V, d).to(device)
U = nn.Embedding(V, d).to(device)
Q = nn.Linear(d, d).to(device)
K = nn.Linear(d, d).to(device)
V = nn.Linear(d, d).to(device)
W1 = nn.Linear(d, d).to(device) # ?
W2 = nn.Linear(d, d).to(device) # ?

optimizer = optim.Adam(
    list(E.parameters() ) +
    list(U.parameters() ) +
    list(Q.parameters() ) +
    list(K.parameters() ) +
    list(V.parameters() ) +
    list(W1.parameters()) +
    list(W2.parameters()),
    lr=lr
)

for epoch in range(epochs):
    total_loss = 0.0
    log_loss = 0.0
    log_steps = 0
    pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch + 1}/{epochs}", unit="step")
    for step in pbar:
        global_step = epoch * steps_per_epoch + step
        current_lr = lr * (1.0 - (global_step / (epochs * steps_per_epoch)))
        for group in optimizer.param_groups: # I'm unsure what this group thing is
            group["lr"] = current_lr
        optimizer.zero_grad()



        loss = F.cross_entropy() # TODO
        loss.backward()
        optimizer.step()
        loss_val = float(loss.detach())
        total_loss += loss_val
        log_loss += loss_val
        log_steps += 1

        log_every = 1_000
        if (step + 1) % log_every == 0:
            pbar.set_postfix(recent_loss=f"{log_loss / log_steps:.4f}")
            log_loss = 0.0
            log_steps = 0
# keep this here do not indent!!!
tqdm.write(f"saved: {checkpoint_path}")
