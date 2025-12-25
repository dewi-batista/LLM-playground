from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

HERE = Path(__file__).resolve().parent

# have read recently about including "mps" for applie products but my m1 macbook air blows lol
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

# model hyperparams
batch_size = 8_192
d = int(config["model"]["d_model"])
epochs = 20
k = 5
lr = 1e-3
steps_per_epoch = 100_000
window = 3

# other hyperparams
log_every = 1_000
V = len(vocab)

# config for storing model params
models_dir = HERE / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# Set RESUME_FROM to None for a new model, "latest" to resume form most recent
# checkpoint and to a given checkpoint path to resume training from said state.
RESUME_FROM = "latest"

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

# for probabilities pertaining to negative indices
neg_probs = np.zeros(V, dtype=np.float64)
for word, info in vocab.items():
    neg_probs[int(info["index"])] = float(info["neg_prob"])
neg_probs /= neg_probs.sum()

# TODO: subsampling frequent words + randomize window size per center (both
# help a lot on text8)
def neg_sampling_loss(u, v, U):
    pos_scores = torch.sum(u * v, dim=1)  # (B,)
    loss = -F.logsigmoid(pos_scores)  # (B,)

    # TODO: ensure that neg_indeces doesn't include the index of the context word
    # TODO: sample using torch.multinomial to remain on GPU
    neg_indeces = np.random.choice(V, size=(u.shape[0], k), p=neg_probs)
    neg_indeces = torch.as_tensor(neg_indeces, dtype=torch.long, device=u.device)  # (B, k)
    neg_vecs = U(neg_indeces)  # (B, k, d)
    neg_scores = torch.sum(neg_vecs * u.unsqueeze(1), dim=2)  # (B, k)
    loss -= torch.sum(F.logsigmoid(-neg_scores), dim=1)  # (B,)

    return torch.mean(loss)

# to go straight from index in text8.txt to token index
indeces_corpus_to_token = np.array(
    [int(vocab[word]["index"]) for word in corpus_text], dtype=np.int32
)
corpus_len = len(indeces_corpus_to_token)

# NOTE: Epochs aren't actually needed here as there's no fixed dataset to
# iterate over. What's implemented is entirely equivalent to instead performing
# epochs * steps_per_epoch optimisation steps. Despite this, epochs are kept
# for the convenience checkpointing loss, parameters, etc.

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

offset_choices = np.array([i for i in range(-window, window + 1) if i != 0], dtype=np.int32)
for epoch in range(start_epoch, epochs):
    total_loss = 0.0
    pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch + 1}/{epochs}", unit="step")
    for step in pbar:
        
        # sample the center word and context word
        idx_centers = np.random.randint(window, corpus_len - window, size=batch_size)
        offsets = np.random.choice(offset_choices, size=batch_size)
        idx_contexts = idx_centers + offsets
        center_idx = torch.as_tensor(indeces_corpus_to_token[idx_centers], dtype=torch.long, device=device)
        context_idx = torch.as_tensor(indeces_corpus_to_token[idx_contexts], dtype=torch.long, device=device)

        optimizer.zero_grad()

        emb_center = E(center_idx)
        emb_contxt = U(context_idx)

        loss = neg_sampling_loss(emb_center, emb_contxt, U)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())

        if log_every > 0 and (step + 1) % log_every == 0:
            pbar.set_postfix(avg_loss=f"{total_loss / (step + 1):.4f}")

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
            "d_model": d,
            "epoch": epoch + 1,
            "rng_state_py": random.getstate(),
            "rng_state_np": np.random.get_state(),
            "rng_state_torch": torch.get_rng_state(),
            "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        checkpoint_path,
    )
    tqdm.write(f"saved: {checkpoint_path}")
