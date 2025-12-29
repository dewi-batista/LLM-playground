# TODO: Include "git yem" in start.sh file

from cache_tokenisation import load_or_create_token_ids
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import json
import numpy as np
import pickle
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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
batch_size = 32
d_model = int(config["model"]["d_model"])
d_ff = 4 * d_model
dropout = 0.1
epochs = 1
lr = 1e-3
min_count = 5
seq_len = 128
steps_per_epoch = 3_000

# prune vocab of sufficiently-infrequent tokens
keep_token_ids = [i for i in range(vocab_size) if int(vocab[str(i)]["count"]) >= min_count]
index_to_token = [vocab[str(i)]["string"] for i in keep_token_ids]
V = len(index_to_token)

token_ids = load_or_create_token_ids(
    language,
    timestamp,
    corpus_path=corpus_path,
    token_ids_path=token_ids_path,
    encodings=encodings,
    vocab=vocab,
)

# map token IDs to embedding indices (and drop pruned tokens)
token_id_to_index = np.full(vocab_size, -1, dtype=np.int32)
for i, token_id in enumerate(keep_token_ids):
    token_id_to_index[token_id] = i
indeces_corpus_to_token = token_id_to_index[token_ids]
indeces_corpus_to_token = indeces_corpus_to_token[indeces_corpus_to_token >= 0]

run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
checkpoint_path = run_dir / f"{language}_transformer_{run_timestamp}.ckpt"
corpus_len = len(indeces_corpus_to_token)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, X):
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        T = X.shape[-2]
        d_model = X.shape[-1]

        M = torch.triu(X.new_full((T, T), float("-inf")), diagonal=1)
        A = torch.softmax((Q @ K.transpose(-2, -1)) / (d_model**0.5) + M, dim=-1)
        O = self.dropout_attn(self.W_O(A @ V))
        H_1 = self.ln1(X + O)

        H_2 = self.dropout_ffn(self.W_2(self.act(self.W_1(H_1))))
        H_3 = self.ln2(H_1 + H_2)
        return H_3

def positional_encoding(seq_len, d_model, device):
    positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1) # (n, 1)

    i = torch.arange(d_model, device=device, dtype=torch.float32)  # (d,)
    div_term = torch.pow(10_000.0, (2 * (i // 2)) / d_model)

    angles = positions / div_term  # (n, d)

    pe = torch.zeros_like(angles)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])

    return pe

dropout_embed = nn.Dropout(dropout).to(device)
E = nn.Embedding(V, d_model).to(device)
final_lay_norm = nn.LayerNorm(d_model).to(device)
model = TransformerBlock(d_model, d_ff, dropout).to(device)
U = nn.Linear(d_model, V, bias=False).to(device)

# TODO: check that this is what it's intended to be (weight tying)
U.weight = E.weight

pe = positional_encoding(seq_len, d_model, device=device)

# NOTE: Does not include U.parameters() due to weight tying
params = (
    list(E.parameters()) +
    list(final_lay_norm.parameters()) +
    list(model.parameters())
)
optimizer = torch.optim.Adam(params, lr=lr)

for epoch in range(epochs):
    log_loss = 0.0
    log_steps = 0
    pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch + 1}/{epochs}", unit="step")
    for step in pbar:
        global_step = epoch * steps_per_epoch + step
        current_lr = lr * (1.0 - (global_step / (epochs * steps_per_epoch)))
        for group in optimizer.param_groups:
            group["lr"] = current_lr # update learning rate of all params
        
        optimizer.zero_grad()

        # sample batch of token sequences (t_i, ..., t_{i + seq_len - 1})
        window_start_idx = np.random.randint(0, corpus_len - seq_len - 1, size=batch_size)
        offsets = np.arange(seq_len, dtype=np.int64)

        context_window = indeces_corpus_to_token[window_start_idx[:, None] + offsets[None, :]]
        targets = indeces_corpus_to_token[window_start_idx[:, None] + offsets[None, :] + 1]

        context_window = torch.as_tensor(context_window, dtype=torch.long, device=device)
        targets = torch.as_tensor(targets, dtype=torch.long, device=device)

        X = dropout_embed(E(context_window) + pe)
        logits = U(final_lay_norm(model(X)))

        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach())
        log_loss += loss_val
        log_steps += 1

        log_every = 1_000
        if (step + 1) % log_every == 0:
            pbar.set_postfix(recent_loss=f"{log_loss / log_steps:.4f}")
            log_loss = 0.0
            log_steps = 0
    # TODO: Check if this is saving for dentical continuation.
    torch.save(
        {
            "E_state_dict": E.state_dict(),
            "model_state_dict": model.state_dict(),
            "final_lay_norm_state_dict": final_lay_norm.state_dict(),
            "U_state_dict": U.state_dict(),
            "vocab_size": V,
            "min_count": min_count,
            "index_to_token": index_to_token,
            "d_model": d_model,
            "d_ff": d_ff,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "dropout": dropout,
            "bpe_vocab_path": str(vocab_path.relative_to(HERE)),
            "bpe_encodings_path": str(encodings_path.relative_to(HERE)),
            "epoch": epoch + 1,
        },
        checkpoint_path,
    )
# keep this here do not indent!!!
tqdm.write(f"saved: {checkpoint_path}")
