from itertools import pairwise
from math import cos, sin
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
sequence_len = 8
d = int(config["model"]["d_model"])
d_ff = 64
epochs = 1
lr = 1e-3
min_count = 5
steps_per_epoch = 1

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

# TODO: make this caching its own script
# load cached tokenisation of corpus into token IDs
token_ids = np.load(token_ids_path, mmap_mode="r")

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

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.act = nn.GELU()

    def forward(self, X):
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        M = torch.triu(torch.full((sequence_len, sequence_len), float('-inf')), diagonal=1).to(device)
        A = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(d) + M, dim=-1)
        O = A @ V
        H_1 = self.ln1(O + X)

        H_2 = self.W_2(self.act(self.W_1(H_1)))
        H_3 = self.ln2(H_2 + H_1)
        return H_3

# TODO: parallelise to output (n x d) instead of (1 x d) now
def positional_encoding(position):
    positions = []
    for i in range(d):
        to_be_sinusoided = position / pow(10_000, 2 * (i // 2) / d)
        if i % 2 == 0:
            positions.append(sin(to_be_sinusoided))
        else:
            positions.append(cos(to_be_sinusoided))
    return positions

E = nn.Embedding(V, d).to(device)
model = TransformerBlock(d, d_ff).to(device)
U = nn.Linear(d, V, bias=False).to(device)

params = (
    list(model.parameters()) +
    list(E.parameters()    ) +
    list(U.parameters()    )
)
optimizer = torch.optim.Adam(params, lr=lr)

for epoch in range(epochs):
    log_loss = 0.0
    log_steps = 0
    # model.train()
    pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch + 1}/{epochs}", unit="step")
    total_loss = 0.0
    for step in pbar:
        global_step = epoch * steps_per_epoch + step
        current_lr = lr * (1.0 - (global_step / (epochs * steps_per_epoch)))
        
        # TODO: understand this group stuff
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        
        optimizer.zero_grad()

        window_start_idx = np.random.randint(1, corpus_len - sequence_len)
        context_window = token_ids[window_start_idx: window_start_idx + sequence_len]
        context_window = torch.tensor(context_window, dtype=torch.int32).to(device)

        X = E(context_window) # TODO: + P(context_window)
        logits = U(model(X))

        print(logits.view(-1, V))

        loss = cross_entropy(logits.view(-1, V), targets.view(-1))
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
