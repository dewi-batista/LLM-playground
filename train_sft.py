# NOTE: This is written assuming that a CUDA device is available.
# NOTE: Most of this code is copy-pasted from train_transformer.py.

from torch.utils.checkpoint import checkpoint
from tfs_utils.core import TransformerBlock, build_token_id_to_index, iter_pre_tokens, make_bpe_encoder, positional_encoding
from tfs_utils.metrics import append_metrics_row, atomic_text_save, write_val_ppl_svg
from tfs_utils.checkpointing import atomic_torch_save

from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

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
if len(sys.argv) < 4 or sys.argv[1] in {"-h", "--help"}:
    print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp> <base_model_number> [sft_run_number]\n")
    raise SystemExit(1)
args = sys.argv[1:]
language = args[0]
timestamp = args[1]
base_model_number = int(args[2])
model_number = int(args[3]) if len(args) > 3 else None

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
base_run_dir = run_dir / f"training_run_{base_model_number}"
base_ckpt_path = base_run_dir / "weights.ckpt"

# new ckpt if model number not passed as argument
if model_number is None:
    model_number = len([p for p in base_run_dir.glob("sft_run_*") if p.is_dir()]) + 1
    resume = False
else:
    resume = True

sft_run_dir = base_run_dir / f"sft_run_{model_number}"
sft_run_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = sft_run_dir / "weights.ckpt"
meta_path = sft_run_dir / "meta.json"
metrics_path = sft_run_dir / "metrics.csv"
val_ppl_plot_path = sft_run_dir / "val_ppl.svg"
tqdm.write(f"\nsft_run_dir: {os.path.relpath(sft_run_dir, HERE)} (resume: {resume})")

config_path = HERE / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

cfg = config["sft"]
batch_size       = int(cfg["batch_size"])
seq_len          = int(cfg["seq_len"])
dropout          = float(cfg["dropout"])
early_stop_delta = float(cfg.get("early_stop_delta", 0.0))
early_stop_pat   = int(cfg.get("early_stop_pat", 0))
eval_batches     = int(cfg["eval_batches"])
eval_every       = int(cfg["eval_every"])
grad_accum_steps = int(cfg["grad_accum_steps"])
grad_checkpoint  = bool(cfg["grad_checkpoint"])
grad_clip        = float(cfg["grad_clip"])
lr               = float(cfg["lr"])
train_tokens     = float(cfg["train_tokens"])
val_frac         = float(cfg["val_frac"])
warmup_frac      = float(cfg["warmup_frac"])
weight_decay     = float(cfg["weight_decay"])

# NOTE on gradient accumulation: Forward and backprop for K micro-batches then
# take the mean in performing the optimisation step.

# NOTE on gradient clipping: If vector of gradient exceeds tau (e.g. tau = 1)
# then resize params relevant to update (recall dropout) to length tau.

# NOTE on weight decay: Just L2 reg. and not applied to bias/LN. Matrix weights
# are what risk overfitting, not biases or LN parameters.

# sanity checks
tqdm.write(f"\n\navailable: {device} (cuda={torch.cuda.is_available()}, cuda_devices={torch.cuda.device_count()})")
tqdm.write(f"device: {torch.cuda.get_device_name(0)}")
tqdm.write(f"amp: {'bf16' if amp_enabled else 'off'}")
tqdm.write(f"tf32: {'on' if torch.backends.cuda.matmul.allow_tf32 else 'off'}")

tqdm.write(f"\nrun_dir: {os.path.relpath(run_dir, HERE)}")
tqdm.write(f"base_run_dir: {os.path.relpath(base_run_dir, HERE)}")
tqdm.write(f"config_path: {os.path.relpath(config_path, HERE)}")
tqdm.write(f"base_ckpt: {os.path.relpath(base_ckpt_path, HERE)}")
tqdm.write(f"sft_ckpt : {os.path.relpath(checkpoint_path, HERE)}")

base_ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
vocab_path = HERE / base_ckpt["bpe_vocab_path"]
encodings_path = HERE / base_ckpt["bpe_encodings_path"]

with open(vocab_path) as f:
    vocab = json.load(f)
with open(encodings_path, "rb") as f:
    encodings = pickle.load(f)
bpe_encode = make_bpe_encoder(encodings)

index_to_token = base_ckpt["index_to_token"]
token_id_to_index, _token_str_to_index = build_token_id_to_index(vocab, index_to_token)
token_id_to_index = token_id_to_index.numpy()

V = len(index_to_token)
d_model = int(base_ckpt["d_model"])
num_heads = int(base_ckpt["num_heads"])
num_blocks = int(base_ckpt["num_blocks"])
d_ff = int(base_ckpt["d_ff"])
tqdm.write(f"arch: V={V}, d_model={d_model}, blocks={num_blocks}, heads={num_heads}, seq_len={seq_len}")

def encode_with_mask(prompt_text: str, full_text: str):
    prompt_ids = []
    for tok in iter_pre_tokens(prompt_text):
        prompt_ids.extend(bpe_encode(tok))
    full_ids = []
    for tok in iter_pre_tokens(full_text):
        full_ids.extend(bpe_encode(tok))

    boundary = len(prompt_ids)
    ids = np.asarray(full_ids, dtype=np.int32)
    mask = np.zeros(len(ids), dtype=np.bool_)
    mask[boundary:] = True

    idx = token_id_to_index[ids]
    keep = idx >= 0
    return idx[keep].astype(np.int32), mask[keep]

tqdm.write("\nloading dataset: squad_v2")
ds = load_dataset("squad_v2", split="train")
ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0 and bool(x["question"]))
ds = ds.map(
    lambda x: {"instruction": x["question"].strip(), "output": x["answers"]["text"][0].strip()},
    remove_columns=ds.column_names,
)
ds = ds.filter(lambda x: bool(x["instruction"]) and bool(x["output"]))
ds = ds.train_test_split(test_size=val_frac, seed=0)
ds_train = ds["train"]
ds_val = ds["test"]

tqdm.write(f"examples: train={len(ds_train):_}, val={len(ds_val):_}")

def build_stream(data):
    all_ids = []
    all_mask = []
    for ex in tqdm(data, desc="tokenising", unit="ex"):
        instruction = ex["instruction"].strip()
        output = ex["output"].strip()
        prompt = f"Instruction: {instruction} Response:"
        full = f"{prompt} {output}"
        ids, mask = encode_with_mask(prompt, full)
        if len(ids) == 0:
            continue
        all_ids.append(ids)
        all_mask.append(mask)
    return np.concatenate(all_ids), np.concatenate(all_mask)

train_ids, train_mask = build_stream(ds_train)
val_ids, val_mask = build_stream(ds_val)
tqdm.write(f"tokens: train={len(train_ids):_}, val={len(val_ids):_}")

# step count computations
tokens_per_step = batch_size * seq_len * grad_accum_steps
total_steps = int(math.ceil(train_tokens / tokens_per_step))
warmup_steps = max(1, int(total_steps * warmup_frac))
tqdm.write(f"\ntrain_tokens: {int(train_tokens):_}\ntokens_per_step: {tokens_per_step:_}\ntotal_steps: {total_steps:_}\nwarmup_steps: {warmup_steps:_}")

# the model begins...
dropout_embed = nn.Dropout(dropout).to(device)
E = nn.Embedding(V, d_model).to(device)
final_lay_norm = nn.LayerNorm(d_model).to(device)
model = nn.Sequential(*[TransformerBlock(d_model, d_ff, num_heads, dropout) for _ in range(num_blocks)]).to(device)
U = nn.Linear(d_model, V, bias=False).to(device)

# weight tying
U.weight = E.weight

E.load_state_dict(base_ckpt["E_state_dict"])
model.load_state_dict(base_ckpt["model_state_dict"])
final_lay_norm.load_state_dict(base_ckpt["final_lay_norm_state_dict"])

# NOTE: Does not include U.parameters() due to weight tying.
params = list(E.parameters()) + list(model.parameters()) + list(final_lay_norm.parameters())

# parameter decay and optimiser (fused Adam for now)
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
    fused=True, # assumes CUDA device + recent PyTorch version
)

start_step = 0
best_val_ppl = float("inf")
patience_count = 0
if resume:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    E.load_state_dict(ckpt["E_state_dict"])

    # NOTE: This checks if the checkpoint was saved with torch.compile(model)
    # in use. If so then remove prefixes.
    model_state_dict = ckpt["model_state_dict"]
    if len(model_state_dict) > 0 and next(iter(model_state_dict.keys())).startswith("_orig_mod."):
        model_state_dict = {k.removeprefix("_orig_mod."): v for k, v in model_state_dict.items()}

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

def batch_loss(token_ids, token_mask):
    starts = np.random.randint(0, len(token_ids) - seq_len - 2, size=batch_size)
    x_np = token_ids[starts[:, None] + offsets[None, :]]
    y_np = token_ids[starts[:, None] + offsets[None, :] + 1]
    m_np = token_mask[starts[:, None] + offsets[None, :] + 1]

    x = torch.as_tensor(x_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    m = torch.as_tensor(m_np, dtype=torch.bool, device=device)

    y = y.masked_fill(~m, -100)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        X = dropout_embed(E(x) + pos_embedding)
        logits = U(final_lay_norm(run_model(X)))
        loss_sum = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=-100, reduction="sum")
    denom = m.sum().clamp(min=1)
    return loss_sum / denom


def eval_nll(token_ids, token_mask, desc):
    total_loss = 0.0
    for _ in tqdm(range(eval_batches), desc=desc, unit="batch", leave=False):
        total_loss += float(batch_loss(token_ids, token_mask))
    return total_loss / eval_batches

if not grad_checkpoint:
    model = torch.compile(model, mode="reduce-overhead")
offsets = np.arange(seq_len, dtype=np.int64)
pos_embedding = positional_encoding(seq_len, d_model, device=device)

def run_model(X):
    if (not grad_checkpoint) or (not torch.is_grad_enabled()):
        return model(X)
    for block in model:
        X = checkpoint(block, X, use_reentrant=False)
    return X

pbar = tqdm(range(start_step, total_steps), desc="Train (SFT)", unit=" batch", total=total_steps, initial=start_step)
train_nll = None
val_nll = None
early_stopped = False
for step in pbar:
    # update lr of all parmams: warmup -> cosine decay
    if step < warmup_steps:
        current_lr = lr * (step + 1) / warmup_steps
    else:
        # cosine decay is: slow (early) -> aggerssive (mid) -> slow (late)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        current_lr = (1.0 + math.cos(math.pi * progress)) * (lr / 2)
    for group in optimizer.param_groups:
        group["lr"] = current_lr
    optimizer.zero_grad()

    for _ in range(grad_accum_steps):
        loss = batch_loss(train_ids, train_mask)
        (loss / grad_accum_steps).backward()
    torch.nn.utils.clip_grad_norm_(params, grad_clip)
    optimizer.step()

    # checkpoint via validation perplexity (over response tokens)
    if (step + 1) % eval_every == 0 or (step + 1) == total_steps:
        E.eval()
        model.eval()
        final_lay_norm.eval()
        U.eval()
        dropout_embed.eval()

        with torch.inference_mode():
            train_nll = eval_nll(train_ids, train_mask, "eval train")
            val_nll = eval_nll(val_ids, val_mask, "eval val")

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
                "min_count": int(base_ckpt.get("min_count", 0)),
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
                "global_step": step + 1,
                "rng_state_py": random.getstate(),
                "rng_state_np": np.random.get_state(),
                "rng_state_torch": torch.get_rng_state(),
                "rng_state_cuda": torch.cuda.get_rng_state_all(),
            }
            save_ok = atomic_torch_save(ckpt_obj, checkpoint_path)
            if save_ok:
                best_val_ppl = val_ppl
                meta = {
                    "stage": "sft",
                    "dataset": "squad_v2",
                    "language": language,
                    "timestamp": timestamp,
                    "base_model_number": base_model_number,
                    "model_number": model_number,
                    "global_step": step + 1,
                    "train_nll": train_nll,
                    "val_nll": val_nll,
                    "val_ppl": val_ppl,
                    "best_val_ppl": val_ppl,
                    "seen_tokens": int((step + 1) * tokens_per_step),
                    "tokens_per_step": tokens_per_step,
                    "total_steps": total_steps,
                    "warmup_steps": warmup_steps,
                    "checkpoint_path": str(checkpoint_path.relative_to(HERE)),
                }
                atomic_text_save(json.dumps(meta, indent=2) + "\n", meta_path)

        improvement = prev_best_val_ppl - val_ppl
        should_stop = False
        if early_stop_pat > 0:
            if improvement >= early_stop_delta:
                patience_count = 0
            else:
                patience_count += 1
            should_stop = (patience_count >= early_stop_pat)
        elif 0 < improvement <= early_stop_delta:
            should_stop = True

        append_metrics_row(
            metrics_path,
            {
                "global_step": step + 1,
                "seen_tokens": int((step + 1) * tokens_per_step),
                "lr": current_lr,
                "recent_loss": train_nll,
                "val_ppl": val_ppl,
                "best_val_ppl": best_val_ppl,
                "patience_count": patience_count,
            },
        )
        write_val_ppl_svg(metrics_path, val_ppl_plot_path)

        pbar.set_postfix(train_nll=f"{train_nll:.3f}", val_nll=f"{val_nll:.3f}")
        if should_stop:
            early_stopped = True
            break
tqdm.write("Supervised fine-tuning complete! (stopped early)" if early_stopped else "Supervised fine-tuning complete!")
