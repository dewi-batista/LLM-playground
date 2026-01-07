# NOTE: This is written assuming that a CUDA device is available.

from cache_tokenisation import load_or_create_token_ids
from tfs_utils.core import TransformerBlock, positional_encoding
from tfs_utils.metrics import append_metrics_row, write_val_ppl_svg

from pathlib import Path
from tqdm import tqdm

import inspect
import json
import math
import numpy as np
import os
import pickle
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# CLI-related
if len(sys.argv) < 3 or sys.argv[1] in {"-h", "--help"}:
    print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp> [model_number]\n")
    raise SystemExit(1)
args = sys.argv[1:]
language = args[0]
timestamp = args[1]
model_number = int(args[2]) if len(args) > 2 else None

# device-related
device = torch.device("cuda")
amp_enabled = (device.type == "cuda")
amp_dtype = torch.bfloat16
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# directory-related
HERE = Path(__file__).resolve().parent

run_dir = HERE / "models" / language / timestamp
run_dir.mkdir(parents=True, exist_ok=True)

config_path = HERE / "config.yaml"
corpus_path = HERE / "data" / f"{language}.txt"
vocab_path = run_dir / "vocabulary.json"
encodings_path = run_dir / "merges.pkl"
token_ids_path = run_dir / "token_ids.npy"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)
with open(encodings_path, "rb") as f:
    encodings = pickle.load(f)
with open(vocab_path) as f:
    vocab = json.load(f)
vocab_size = len(vocab)

# new training run if model_number not passed as argument
if model_number is None:
    model_number = len([p for p in run_dir.glob("training_run_*") if p.is_dir()]) + 1
    resume = False
else:
    resume = True

#
training_run_dir = run_dir / f"training_run_{model_number}"
training_run_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = training_run_dir / "weights.ckpt"
meta_path = training_run_dir / "meta.json"
metrics_path = training_run_dir / "metrics.csv"
val_ppl_plot_path = training_run_dir / "val_ppl.svg"
tqdm.write(f"\ntraining_run_dir: {os.path.relpath(training_run_dir, HERE)} (resume: {resume})")

# sanity checks
tqdm.write(f"\n\navailable: {device} (cuda={torch.cuda.is_available()}, cuda_devices={torch.cuda.device_count()})")
tqdm.write(f"device: {torch.cuda.get_device_name(0)}")
tqdm.write(f"amp: {'bf16' if amp_enabled else 'off'}")
tqdm.write(f"tf32: {'on' if (device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32) else 'off'}")

tqdm.write(f"\nrun_dir: {os.path.relpath(run_dir, HERE)}")
tqdm.write(f"config_path: {os.path.relpath(config_path, HERE)}")
tqdm.write(f"corpus_path: {os.path.relpath(corpus_path, HERE)} (exists={corpus_path.exists()})")
tqdm.write(f"encodings_path: {os.path.relpath(encodings_path, HERE)} (exists={encodings_path.exists()})")
tqdm.write(f"vocab_path: {os.path.relpath(vocab_path, HERE)} (exists={vocab_path.exists()})")
tqdm.write(f"token_ids_path: {os.path.relpath(token_ids_path, HERE)} (exists={token_ids_path.exists()})")
tqdm.write(f"size of token_ids: {token_ids_path.stat().st_size / (1024**2):.1f} MB")

tqdm.write(f"\nnum mergers (encodings): {len(encodings):_}")
tqdm.write(f"num tokens in vocab: {vocab_size:_}")

# hyperparams
cfg = config["transformer"]

batch_size       = int(cfg["batch_size"])
d_model          = int(cfg["d_model"])
dropout          = float(cfg["dropout"])
early_stop_delta = float(cfg["early_stop_delta"])
early_stop_pat   = int(cfg["early_stop_pat"])
eval_batches     = int(cfg["eval_batches"])
eval_every       = int(cfg["eval_every"])
grad_accum_steps = int(cfg["grad_accum_steps"])
grad_clip        = float(cfg["grad_clip"])
lr               = float(cfg["lr"])
min_count        = int(cfg["min_count"])
num_blocks       = int(cfg["num_blocks"])
seq_len          = int(cfg["seq_len"])
train_tokens     = float(cfg["train_tokens"])
val_frac         = float(cfg["val_frac"])
warmup_frac      = float(cfg["warmup_frac"])
weight_decay     = float(cfg["weight_decay"])

# dependent hyperparams
d_ff = 4 * d_model
num_heads = d_model // 64

# NOTE on gradient accumulation: Forward and backprop for K micro-batches then
# take the mean in performing the optimisation step.

# NOTE on gradient clipping: If vector of gradient exceeds tau (e.g. tau = 1)
# then resize params relevant to update (recall dropout) to length tau.

# NOTE on weight decay: Just L2 reg. and not applied to bias/LN. Matrix weights
# are what risk overfitting, not biases or LN parameters.

# prune vocab of sufficiently-infrequent tokens
keep_token_ids = [i for i in range(vocab_size) if int(vocab[str(i)]["count"]) >= min_count]
index_to_token = [vocab[str(i)]["string"] for i in keep_token_ids]
V = len(index_to_token) # TODO: Understand discrepancy between vocab_size and V.

token_ids = load_or_create_token_ids(
    language,
    timestamp,
    corpus_path=corpus_path,
    token_ids_path=token_ids_path,
    encodings=encodings,
    vocab=vocab,
)

# map token IDs to embedding indices
token_id_to_index = np.full(vocab_size, -1, dtype=np.int32)
for i, token_id in enumerate(keep_token_ids):
    token_id_to_index[token_id] = i
indeces_corpus_to_token = token_id_to_index[token_ids]
indeces_corpus_to_token = indeces_corpus_to_token[indeces_corpus_to_token >= 0]

# step count computations
tokens_per_step = batch_size * seq_len * grad_accum_steps
total_steps = int(math.ceil(train_tokens / tokens_per_step))
warmup_steps = max(1, int(total_steps * warmup_frac))

# split corpus into training and validation
corpus_len = len(indeces_corpus_to_token)
val_start = int(corpus_len * (1 - val_frac))
train_token_ids = indeces_corpus_to_token[:val_start]
val_token_ids = indeces_corpus_to_token[val_start:]

# additional sanity checks
tqdm.write(f"built indeces_corpus_to_token (length: {len(indeces_corpus_to_token):_})")

tqdm.write(f"\ntrain_tokens: {int(train_tokens):_}\ntokens_per_step: {tokens_per_step:_}\ntotal_steps: {total_steps:_}\nwarmup_steps: {warmup_steps:_}")
tqdm.write(f"early_stop: delta={early_stop_delta}, patience={early_stop_pat}")

tqdm.write(f"\ntrain_len: {len(train_token_ids):_}, val_len: {len(val_token_ids):_}")

# the model begins...
dropout_embed = nn.Dropout(dropout).to(device)
E = nn.Embedding(V, d_model).to(device)
final_lay_norm = nn.LayerNorm(d_model).to(device)
model = nn.Sequential(*[TransformerBlock(d_model, d_ff, num_heads, dropout) for _ in range(num_blocks)]).to(device)
U = nn.Linear(d_model, V, bias=False).to(device)

# GPT-2-style weight-init (suggested to me) for stability
# W elements ~ N(0, 0.02), biases = 0, layer norm scale = 1, shift = 0
# Why 0.02? In the words of Neel Nanda: who knows, it works.
def init_gpt2(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

if not resume:
    E.apply(init_gpt2)
    model.apply(init_gpt2)
    final_lay_norm.apply(init_gpt2)

    resid_scale = math.sqrt(2 * num_blocks)
    for block in model:
        block.W_O.weight.data /= resid_scale
        block.W_2.weight.data /= resid_scale

# weight tying
U.weight = E.weight

# NOTE: Does not include U.parameters() due to weight tying.
params = list(E.parameters()) + list(model.parameters()) + list(final_lay_norm.parameters())

# parameter decay and optimiser (fused Adam for now)
decay_params = [p for p in params if p.ndim >= 2]
non_decay_params = [p for p in params if p.ndim < 2]
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": non_decay_params, "weight_decay": 0.0},
    ],
    lr=lr,
    betas=(0.9, 0.95),
    eps=1e-8,
    fused=True, # assumes CUDA device + recent PyTorch version
)

# NOTE: The if statement inside checks if the checkpoint was saved with
# torch.compile(model) in use. If so then the relevant prefixes are removed.
best_val_ppl = float("inf")
start_step = 0
if resume:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state_dict = ckpt["model_state_dict"]
    if len(model_state_dict) > 0 and next(iter(model_state_dict.keys())).startswith("_orig_mod."):
        model_state_dict = {k.removeprefix("_orig_mod."): v for k, v in model_state_dict.items()}

    E.load_state_dict(ckpt["E_state_dict"])
    model.load_state_dict(model_state_dict)
    final_lay_norm.load_state_dict(ckpt["final_lay_norm_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = int(ckpt.get("global_step", 0))
    best_val_ppl = float(ckpt.get("best_val_ppl", float("inf")))
    random.setstate(ckpt["rng_state_py"])
    np.random.set_state(ckpt["rng_state_np"])
    torch.set_rng_state(ckpt["rng_state_torch"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng_state_cuda"]])
    
    tqdm.write(f"\nresuming from: {os.path.relpath(checkpoint_path, HERE)}\n(start step: {start_step:_})")

def eval_nll(token_ids): # eval NLL on random batches (dropout off)
    total_loss = 0.0
    for _ in range(eval_batches):
        window_start_idx = np.random.randint(0, len(token_ids) - seq_len - 1, size=batch_size)
        context_window = token_ids[window_start_idx[:, None] + offsets[None, :]]
        targets = token_ids[window_start_idx[:, None] + offsets[None, :] + 1]

        context_window = torch.as_tensor(context_window, dtype=torch.long, device=device)
        targets = torch.as_tensor(targets, dtype=torch.long, device=device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            X = dropout_embed(E(context_window) + pos_embedding)
            logits = U(final_lay_norm(model(X)))
            loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        total_loss += float(loss)

    return total_loss / eval_batches

model = torch.compile(model, mode="reduce-overhead")
offsets = np.arange(seq_len, dtype=np.int64)
pos_embedding = positional_encoding(seq_len, d_model, device=device)

early_stopped = False
no_improve_evals = 0
pbar = tqdm(range(start_step, total_steps), desc="Train", unit=" batch", total=total_steps, initial=start_step)
train_nll = None
val_nll = None

for step in pbar:

    # update lr of all parmams: warmup -> cosine decay
    if step < warmup_steps:
        current_lr = lr * (step + 1) / warmup_steps
    else:
        # cosine decay is: slow (early) -> aggerssive (mid) -> slow (late)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        current_lr = (1.0 + math.cos(math.pi * progress)) * (lr / 2)
        # tqdm.write(f"I am invoked, {lr}, {step}, {warmup_steps}, {total_steps}, {progress},{current_lr}")
    for group in optimizer.param_groups:
        group["lr"] = current_lr
    optimizer.zero_grad()

    for _ in range(grad_accum_steps):
        # sample batch of token sequences (t_i, ..., t_{i + seq_len - 1})
        window_start_idx = np.random.randint(0, len(train_token_ids) - seq_len - 1, size=batch_size)
        context_window = train_token_ids[window_start_idx[:, None] + offsets[None, :]]
        targets = train_token_ids[window_start_idx[:, None] + offsets[None, :] + 1]

        context_window = torch.as_tensor(context_window, dtype=torch.long, device=device)
        targets = torch.as_tensor(targets, dtype=torch.long, device=device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            X = dropout_embed(E(context_window) + pos_embedding)
            logits = U(final_lay_norm(model(X)))
            loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        (loss / grad_accum_steps).backward()
    torch.nn.utils.clip_grad_norm_(params, grad_clip)
    optimizer.step()

    # checkpoint via validation perplexity
    if (step + 1) % eval_every == 0 or (step + 1) == total_steps:
        E.eval()
        model.eval()
        final_lay_norm.eval()
        U.eval()
        dropout_embed.eval()

        with torch.inference_mode():
            train_nll = eval_nll(train_token_ids)
            val_nll = eval_nll(val_token_ids)

        E.train()
        model.train()
        final_lay_norm.train()
        U.train()
        dropout_embed.train()

        val_ppl = math.exp(val_nll)
        prev_best_val_ppl = best_val_ppl
        is_best = (val_ppl < prev_best_val_ppl)

        if is_best:
            model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt_obj = {
                "E_state_dict": E.state_dict(),
                "model_state_dict": model_to_save.state_dict(),
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
                "best_val_ppl": val_ppl,
                "val_ppl": val_ppl,
                "bpe_vocab_path": str(vocab_path.relative_to(HERE)),
                "bpe_encodings_path": str(encodings_path.relative_to(HERE)),
                "token_ids_path": str(token_ids_path.relative_to(HERE)),
                "global_step": step + 1,
                "rng_state_py": random.getstate(),
                "rng_state_np": np.random.get_state(),
                "rng_state_torch": torch.get_rng_state(),
                "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            torch.save(ckpt_obj, checkpoint_path)
            best_val_ppl = val_ppl
            meta = {
                "language": language,
                "timestamp": timestamp,
                "model_number": model_number,
                "global_step": step + 1,
                "val_ppl": val_ppl,
                "best_val_ppl": val_ppl,
                "prev_best_val_ppl": prev_best_val_ppl,
                "train_tokens": int(train_tokens),
                "seen_tokens": int((step + 1) * tokens_per_step),
                "tokens_per_step": tokens_per_step,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
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
                "grad_accum_steps": grad_accum_steps,
                "val_frac": val_frac,
                "eval_every": eval_every,
                "eval_batches": eval_batches,
                "vocabulary_path": str(vocab_path.relative_to(HERE)),
                "merges_path": str(encodings_path.relative_to(HERE)),
                "token_ids_path": str(token_ids_path.relative_to(HERE)),
                "checkpoint_path": str(checkpoint_path.relative_to(HERE)),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
                f.write("\n")
            tqdm.write(f"saved: {checkpoint_path} (step={step + 1}, val_ppl={val_ppl:.2f})")

        append_metrics_row(
            metrics_path,
            {
                "global_step": step + 1,
                "seen_tokens": int((step + 1) * tokens_per_step),
                "lr": current_lr,
                "recent_loss": train_nll,
                "val_ppl": val_ppl,
                "best_val_ppl": best_val_ppl,
                "patience_count": no_improve_evals,
            },
        )
        write_val_ppl_svg(metrics_path, val_ppl_plot_path)
        pbar.set_postfix(
            train_nll=f"{train_nll:.3f}",
            val_nll=f"{val_nll:.3f}",
        )
        improvement = prev_best_val_ppl - val_ppl
        if early_stop_pat > 0:
            if improvement >= early_stop_delta:
                no_improve_evals = 0
            else:
                no_improve_evals += 1
            if no_improve_evals >= early_stop_pat:
                early_stopped = True
                break
        elif 0 < improvement <= early_stop_delta:
            early_stopped = True
            break
tqdm.write("Training complete! (stopped early)" if early_stopped else "Training complete!")
