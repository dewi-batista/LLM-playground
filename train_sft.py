from pathlib import Path

from datasets import load_dataset
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

from tfs_utils.metrics import append_metrics_row, write_val_ppl_svg
from tfs_utils.core import (
    TransformerBlock,
    build_token_id_to_index,
    iter_pre_tokens,
    make_bpe_encoder,
    positional_encoding,
)

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

if len(sys.argv) < 4 or sys.argv[1] in {"-h", "--help"}:
    print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp> <base_model_number> [sft_model_number]")
    raise SystemExit(1)

args = sys.argv[1:]
language = args[0]
timestamp = args[1]
base_model_number = int(args[2])
sft_model_number = int(args[3]) if len(args) > 3 else None

run_dir = MODELS_DIR / language / timestamp
base_ckpt_path = run_dir / f"training_run_{base_model_number}" / "weights.ckpt"

if sft_model_number is None:
    sft_model_number = len([p for p in run_dir.glob("training_run_*") if p.is_dir()]) + 1

sft_dir = run_dir / f"training_run_{sft_model_number}"
sft_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = sft_dir / "weights.ckpt"
meta_path = sft_dir / "meta.json"
metrics_path = sft_dir / "metrics.csv"
plot_path = sft_dir / "val_ppl.svg"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
amp_enabled = (device.type == "cuda")
amp_dtype = torch.bfloat16
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

config_path = HERE / "config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)["transformer"]

batch_size = int(cfg["batch_size"])
seq_len = int(cfg["seq_len"])
dropout = float(cfg["dropout"])
lr = float(cfg["lr"])
weight_decay = float(cfg["weight_decay"])
grad_clip = float(cfg["grad_clip"])
grad_accum_steps = int(cfg["grad_accum_steps"])
eval_every = int(cfg["eval_every"])
eval_batches = int(cfg["eval_batches"])
val_frac = float(cfg["val_frac"])
warmup_frac = float(cfg["warmup_frac"])
train_tokens = float(cfg["train_tokens"])

tqdm.write(f"device: {device} (amp={'bf16' if amp_enabled else 'off'})")
tqdm.write(f"base_ckpt: {os.path.relpath(base_ckpt_path, HERE)}")
tqdm.write(f"sft_ckpt : {os.path.relpath(checkpoint_path, HERE)}")

ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
vocab_path = Path(ckpt["bpe_vocab_path"])
encodings_path = Path(ckpt["bpe_encodings_path"])

with open(vocab_path) as f:
    vocab = json.load(f)
with open(encodings_path, "rb") as f:
    encodings = pickle.load(f)
bpe_encode = make_bpe_encoder(encodings)

index_to_token = ckpt["index_to_token"]
token_id_to_index, _token_str_to_index = build_token_id_to_index(vocab, index_to_token)
token_id_to_index = token_id_to_index.numpy()

V = len(index_to_token)
d_model = int(ckpt["d_model"])
num_heads = int(ckpt["num_heads"])
num_blocks = int(ckpt["num_blocks"])
d_ff = int(ckpt["d_ff"])
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

tqdm.write("loading dataset: tatsu-lab/alpaca")
ds = load_dataset("tatsu-lab/alpaca")["train"]
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

offsets = np.arange(seq_len, dtype=np.int64)

dropout_embed = nn.Dropout(dropout).to(device)
E = nn.Embedding(V, d_model).to(device)
final_lay_norm = nn.LayerNorm(d_model).to(device)
model = nn.Sequential(*[TransformerBlock(d_model, d_ff, num_heads, dropout) for _ in range(num_blocks)]).to(device)
U = nn.Linear(d_model, V, bias=False).to(device)
U.weight = E.weight

E.load_state_dict(ckpt["E_state_dict"])
model.load_state_dict(ckpt["model_state_dict"])
final_lay_norm.load_state_dict(ckpt["final_lay_norm_state_dict"])

params = list(E.parameters()) + list(model.parameters()) + list(final_lay_norm.parameters())
decay_params = [p for p in params if p.ndim >= 2]
no_decay_params = [p for p in params if p.ndim < 2]
adamw_sig = inspect.signature(torch.optim.AdamW)
fused_adamw = device.type == "cuda" and ("fused" in adamw_sig.parameters)
optimizer_kwargs = {"lr": lr, "betas": (0.9, 0.95), "eps": 1e-8}
if fused_adamw:
    optimizer_kwargs["fused"] = True
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    **optimizer_kwargs,
)

pe = positional_encoding(seq_len, d_model, device=device)

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
        X = dropout_embed(E(x) + pe)
        logits = U(final_lay_norm(model(X)))
        loss_sum = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), ignore_index=-100, reduction="sum")
    denom = m.sum().clamp(min=1)
    return loss_sum / denom

@torch.no_grad()
def eval_nll(token_ids, token_mask):
    E.eval()
    model.eval()
    final_lay_norm.eval()
    U.eval()
    dropout_embed.eval()
    with torch.inference_mode():
        total = 0.0
        for _ in range(eval_batches):
            loss = batch_loss(token_ids, token_mask)
            total += float(loss)
    E.train()
    model.train()
    final_lay_norm.train()
    U.train()
    dropout_embed.train()
    return total / eval_batches

tokens_per_step = batch_size * seq_len * grad_accum_steps
total_steps = int(math.ceil(train_tokens / tokens_per_step))
warmup_steps = max(1, int(total_steps * warmup_frac))
tqdm.write(f"train_tokens={int(train_tokens):_} total_steps={total_steps:_} warmup_steps={warmup_steps:_}")

best_val_ppl = float("inf")
start_step = 0
if checkpoint_path.exists():
    ckpt2 = torch.load(checkpoint_path, map_location=device, weights_only=False)
    E.load_state_dict(ckpt2["E_state_dict"])
    model.load_state_dict(ckpt2["model_state_dict"])
    final_lay_norm.load_state_dict(ckpt2["final_lay_norm_state_dict"])
    optimizer.load_state_dict(ckpt2["optimizer_state_dict"])
    start_step = int(ckpt2.get("global_step", 0))
    best_val_ppl = float(ckpt2.get("best_val_ppl", float("inf")))
    random.setstate(ckpt2["rng_state_py"])
    np.random.set_state(ckpt2["rng_state_np"])
    torch.set_rng_state(ckpt2["rng_state_torch"].cpu())
    if torch.cuda.is_available() and ckpt2["rng_state_cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt2["rng_state_cuda"]])
    tqdm.write(f"resuming from step={start_step:_}")

pbar = tqdm(range(start_step, total_steps), desc="SFT", unit=" batch", total=total_steps, initial=start_step)
for step in pbar:
    if step < warmup_steps:
        current_lr = lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        current_lr = (1.0 + math.cos(math.pi * progress)) * (lr / 2)
    for group in optimizer.param_groups:
        group["lr"] = current_lr
    optimizer.zero_grad()

    step_loss = 0.0
    for _ in range(grad_accum_steps):
        loss = batch_loss(train_ids, train_mask)
        (loss / grad_accum_steps).backward()
        step_loss += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(params, grad_clip)
    optimizer.step()

    if (step + 1) % eval_every == 0 or (step + 1) == total_steps:
        train_nll = eval_nll(train_ids, train_mask)
        val_nll = eval_nll(val_ids, val_mask)

        val_ppl = math.exp(val_nll)
        is_best = (val_ppl < best_val_ppl)

        if is_best:
            ckpt_obj = {
                "E_state_dict": E.state_dict(),
                "model_state_dict": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                "final_lay_norm_state_dict": final_lay_norm.state_dict(),
                "U_state_dict": U.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": V,
                "min_count": int(ckpt.get("min_count", 0)),
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
                "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            torch.save(ckpt_obj, checkpoint_path)
            best_val_ppl = val_ppl
            meta = {
                "stage": "sft",
                "dataset": "tatsu-lab/alpaca",
                "language": language,
                "timestamp": timestamp,
                "base_model_number": base_model_number,
                "model_number": sft_model_number,
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
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
                f.write("\n")

        append_metrics_row(
            metrics_path,
            {
                "global_step": step + 1,
                "seen_tokens": int((step + 1) * tokens_per_step),
                "lr": current_lr,
                "recent_loss": train_nll,
                "val_ppl": val_ppl,
                "best_val_ppl": best_val_ppl,
                "patience_count": 0,
            },
        )
        write_val_ppl_svg(metrics_path, plot_path)

        pbar.set_postfix(train_nll=f"{train_nll:.4f}", val_nll=f"{val_nll:.4f}")

tqdm.write("SFT complete!")
