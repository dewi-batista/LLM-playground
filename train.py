from datetime import datetime
from pathlib import Path
from random import randint
from tqdm import tqdm

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

HERE = Path(__file__).resolve().parent

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# for text
with open(HERE / "./data/text8.txt") as f:
    corpus_text = f.read().split()

# for vocab
with open(HERE / "./data/vocabulary.json") as f:
    vocab = json.load(f)

# for config
with open(HERE / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

# hyperparams
d = int(config["model"]["d_model"])
epochs = 5
k = 5
lr = 1e-3
steps_per_epoch = 100_000
V = len(vocab)
window = 3
log_every = 1000
models_dir = HERE / "models"
models_dir.mkdir(parents=True, exist_ok=True)
run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
checkpoint_path = models_dir / f"{run_timestamp}.ckpt"

# for probabilities pertaining to negative indices
neg_probs = np.zeros(V, dtype=np.float64)
for word, info in vocab.items():
    neg_probs[int(info["index"])] = float(info["neg_prob"])
neg_probs /= neg_probs.sum()

def neg_sampling_loss(u, v, U):
    loss = -F.logsigmoid(torch.dot(u, v))
    
    # TODO: ensure that neg_indeces doesn't include the index of the context word
    neg_indeces = np.random.choice(V, size=k, p=neg_probs)
    for neg_idx in neg_indeces:
        loss -= F.logsigmoid(-torch.dot(u, U(torch.tensor(int(neg_idx), dtype=torch.long, device=device))))
    
    return loss

# to go straight from index in text8.txt to token index
indeces_corpus_to_token = [int(vocab[word]["index"]) for word in corpus_text]
corpus_len = len(indeces_corpus_to_token)

E = nn.Embedding(V, d).to(device)
U = nn.Embedding(V, d).to(device)
optimizer = optim.Adam(list(E.parameters()) + list(U.parameters()), lr=lr)
for epoch in range(epochs):
    total_loss = 0.0
    pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch + 1}/{epochs}", unit="step")
    for step in pbar:
        
        # sample the center word and context word
        idx_center = randint(window, corpus_len - window - 1)
        center_idx = indeces_corpus_to_token[idx_center]

        offset = 0
        while offset == 0:
            offset = randint(-window, window)
        context_idx = indeces_corpus_to_token[idx_center + offset]

        optimizer.zero_grad()

        emb_center = E(torch.tensor(center_idx, dtype=torch.long, device=device))
        emb_contxt = U(torch.tensor(context_idx, dtype=torch.long, device=device))

        loss = neg_sampling_loss(emb_center, emb_contxt, U)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())

        if log_every > 0 and (step + 1) % log_every == 0:
            pbar.set_postfix(avg_loss=f"{total_loss / (step + 1):.4f}")

    avg_loss = total_loss / steps_per_epoch
    tqdm.write(f"epoch {epoch + 1}/{epochs} avg_loss={avg_loss:.4f}")
    torch.save(
        {
            "E_state_dict": E.state_dict(),
            "U_state_dict": U.state_dict(),
            "vocab_size": V,
            "d_model": d,
        },
        checkpoint_path,
    )
    tqdm.write(f"saved: {checkpoint_path}")
