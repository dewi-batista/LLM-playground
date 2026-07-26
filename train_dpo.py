# NOTE: This is written assuming that a CUDA device is available.
# NOTE: Most of this code is copy-pasted from train_sft.py.

from tfs_utils.core import build_token_id_to_index, iter_pre_tokens, make_bpe_encoder
from tfs_utils.model_factory import arch_to_ckpt_fields, build_model, resolve_arch_config
from tfs_utils.metrics import append_metrics_row, atomic_text_save
from tfs_utils.checkpointing import atomic_torch_save
from torch.utils.checkpoint import checkpoint

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
import torch.nn.functional as F
import yaml

# CLI-related
if len(sys.argv) < 5 or sys.argv[1] in {"-h", "--help"}:
    print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp> <base_model_number> <sft_run_number> [dpo_run_number]\n")
    raise SystemExit(1)
args = sys.argv[1:]
language = args[0]
timestamp = args[1]
base_model_number = int(args[2])
sft_run_number = int(args[3])
model_number = int(args[4]) if len(args) > 4 else None

# device-related
device = torch.device("cuda")
amp_enabled = (device.type == "cuda")
amp_dtype = torch.bfloat16
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# directory-related
HERE = Path(__file__).resolve().parent
run_dir = HERE / "models" / language / timestamp
sft_run_dir = run_dir / f"training_run_{base_model_number}" / f"sft_run_{sft_run_number}"
sft_ckpt_path = sft_run_dir / "weights.ckpt"

# new ckpt if model number not passed as argument
if model_number is None:
    model_number = len([p for p in sft_run_dir.glob("dpo_run_*") if p.is_dir()]) + 1
    resume = False
else:
    resume = True

dpo_run_dir = sft_run_dir / f"dpo_run_{model_number}"
dpo_run_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = dpo_run_dir / "weights.ckpt"
meta_path = dpo_run_dir / "meta.json"
metrics_path = dpo_run_dir / "metrics.csv"
tqdm.write(f"\ndpo_run_dir: {os.path.relpath(dpo_run_dir, HERE)} (resume: {resume})")

config_path = HERE / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# NOTE on DPO: unlike pretraining/SFT, which sample sliding windows from a
# continuous token stream, DPO needs each (prompt+response) example kept as a
# discrete, complete unit -- a valid whole-sequence log-prob requires the full
# response, not an arbitrary window into it. So examples are packed to a fixed
# length (seq_len + 1) individually, and batches are whole-example draws, not
# window samples from a flattened corpus.

# sanity checks
tqdm.write(f"\n\navailable: {device} (cuda={torch.cuda.is_available()}, cuda_devices={torch.cuda.device_count()})")
tqdm.write(f"device: {torch.cuda.get_device_name(0)}")
tqdm.write(f"amp: {'bf16' if amp_enabled else 'off'}")
tqdm.write(f"tf32: {'on' if torch.backends.cuda.matmul.allow_tf32 else 'off'}")

tqdm.write(f"\nrun_dir: {os.path.relpath(run_dir, HERE)}")
tqdm.write(f"sft_run_dir: {os.path.relpath(sft_run_dir, HERE)}")
tqdm.write(f"config_path: {os.path.relpath(config_path, HERE)}")
tqdm.write(f"sft_ckpt: {os.path.relpath(sft_ckpt_path, HERE)}")
tqdm.write(f"dpo_ckpt: {os.path.relpath(checkpoint_path, HERE)}")

sft_ckpt = torch.load(sft_ckpt_path, map_location="cpu", weights_only=False)

d_model = int(sft_ckpt["d_model"])
if d_model == 768:
    size = "small"
elif d_model == 1024:
    size = "medium"
elif d_model == 1280:
    size = "large"
else:
    raise SystemExit(f"unknown model size for d_model={d_model}")

cfg = config[f"dpo-{size}"]
batch_size       = int(cfg["batch_size"])
seq_len          = int(cfg["seq_len"])
beta             = float(cfg["beta"])
early_stop_delta = float(cfg.get("early_stop_delta", 0.0))
early_stop_pat   = int(cfg.get("early_stop_pat", 0))
eval_batches     = int(cfg["eval_batches"])
eval_every       = int(cfg["eval_every"])
grad_accum_steps = int(cfg["grad_accum_steps"])
grad_checkpoint  = bool(cfg["grad_checkpoint"])
grad_clip        = float(cfg["grad_clip"])
lr               = float(cfg["lr"])
total_steps      = int(cfg["total_steps"])
val_frac         = float(cfg["val_frac"])
warmup_frac      = float(cfg["warmup_frac"])
weight_decay     = float(cfg["weight_decay"])
warmup_steps     = max(1, int(total_steps * warmup_frac))

vocab_path = HERE / sft_ckpt["bpe_vocab_path"]
encodings_path = HERE / sft_ckpt["bpe_encodings_path"]

with open(vocab_path) as f:
    vocab = json.load(f)
with open(encodings_path, "rb") as f:
    encodings = pickle.load(f)
bpe_encode = make_bpe_encoder(encodings)

index_to_token = sft_ckpt["index_to_token"]
token_id_to_index, _token_str_to_index = build_token_id_to_index(vocab, index_to_token)
token_id_to_index = token_id_to_index.numpy()

V = len(index_to_token)
arch = resolve_arch_config(sft_ckpt, overrides={"seq_len": seq_len, "dropout": 0.0})
num_heads, num_blocks, d_ff = arch.num_heads, arch.num_blocks, arch.d_ff
tqdm.write(f"arch: V={V}, d_model={d_model}, blocks={num_blocks}, heads={num_heads}, seq_len={seq_len}")
tqdm.write(f"\ntotal_steps: {total_steps:_}\nwarmup_steps: {warmup_steps:_}")


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
    prompt_len = int(keep[:boundary].sum())  # prompt length after index filtering
    return idx[keep].astype(np.int32), mask[keep], prompt_len


def pack_example(prompt_text: str, response_text: str):
    full_text = f"{prompt_text} {response_text}"
    ids, mask, prompt_len = encode_with_mask(prompt_text, full_text)
    if prompt_len > seq_len:
        return None  # prompt alone doesn't fit -- skip rather than truncate it away

    target_len = seq_len + 1
    if len(ids) > target_len:
        # truncate only the response tail; prompt_len <= seq_len < target_len
        # so the full prompt always survives this slice.
        ids = ids[:target_len]
        mask = mask[:target_len]
    pad_len = target_len - len(ids)
    if pad_len > 0:
        ids = np.concatenate([ids, np.zeros(pad_len, dtype=np.int32)])
        mask = np.concatenate([mask, np.zeros(pad_len, dtype=np.bool_)])
    return ids, mask


def build_packed(data, desc):
    c_ids_list, c_mask_list, r_ids_list, r_mask_list = [], [], [], []
    skipped = 0
    for ex in tqdm(data, desc=desc, unit="ex"):
        question = ex["question"].strip()
        chosen = ex["chosen"].strip()
        rejected = ex["rejected"].strip()
        prompt = f"Instruction: {question} Response:"

        c = pack_example(prompt, chosen)
        r = pack_example(prompt, rejected)
        if c is None or r is None:
            skipped += 1
            continue
        c_ids_list.append(c[0]); c_mask_list.append(c[1])
        r_ids_list.append(r[0]); r_mask_list.append(r[1])
    tqdm.write(f"{desc}: kept {len(c_ids_list):_}, skipped {skipped:_} (prompt too long)")
    return (
        np.stack(c_ids_list), np.stack(c_mask_list),
        np.stack(r_ids_list), np.stack(r_mask_list),
    )


tqdm.write("\nloading dataset: Intel/orca_dpo_pairs")
ds = load_dataset("Intel/orca_dpo_pairs", split="train")
ds = ds.filter(lambda x: bool(x["question"]) and bool(x["chosen"]) and bool(x["rejected"]))
ds = ds.train_test_split(test_size=val_frac, seed=0)
ds_train = ds["train"]
ds_val = ds["test"]
tqdm.write(f"examples: train={len(ds_train):_}, val={len(ds_val):_}")

train_c_ids, train_c_mask, train_r_ids, train_r_mask = build_packed(ds_train, "tokenising (train)")
val_c_ids, val_c_mask, val_r_ids, val_r_mask = build_packed(ds_val, "tokenising (val)")

# the model begins... two copies: a trainable policy and a frozen reference,
# both initialised from the same SFT checkpoint. dropout=0.0 (forced via the
# ArchConfig override above) is structural, not toggled via .eval()/.train() --
# nn.Dropout(0.0) is a no-op regardless of mode, matching standard DPO practice
# of training with dropout disabled so the chosen/rejected log-prob comparison
# isn't polluted by stochastic noise.
policy_bundle = build_model(arch, V, device)
ref_bundle = build_model(arch, V, device)


def _strip_compile_prefix(state_dict):
    if len(state_dict) > 0 and next(iter(state_dict.keys())).startswith("_orig_mod."):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


sft_model_state_dict = _strip_compile_prefix(sft_ckpt["model_state_dict"])
for bundle_ in (policy_bundle, ref_bundle):
    bundle_.E.load_state_dict(sft_ckpt["E_state_dict"])
    bundle_.model.load_state_dict(sft_model_state_dict)
    bundle_.final_lay_norm.load_state_dict(sft_ckpt["final_lay_norm_state_dict"])

for p in list(ref_bundle.E.parameters()) + list(ref_bundle.model.parameters()) + list(ref_bundle.final_lay_norm.parameters()):
    p.requires_grad_(False)

policy_E, policy_model, policy_final_lay_norm, policy_U = (
    policy_bundle.E, policy_bundle.model, policy_bundle.final_lay_norm, policy_bundle.U
)
ref_E, ref_model, ref_final_lay_norm, ref_U = (
    ref_bundle.E, ref_bundle.model, ref_bundle.final_lay_norm, ref_bundle.U
)
policy_pe, ref_pe = policy_bundle.pe, ref_bundle.pe

# NOTE: Does not include U.parameters() due to weight tying.
params = list(policy_E.parameters()) + list(policy_model.parameters()) + list(policy_final_lay_norm.parameters())

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
    fused=True,  # assumes CUDA device + recent PyTorch version
)

start_step = 0
best_val_loss = float("inf")
patience_count = 0
if resume:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy_E.load_state_dict(ckpt["E_state_dict"])
    policy_model.load_state_dict(_strip_compile_prefix(ckpt["model_state_dict"]))
    policy_final_lay_norm.load_state_dict(ckpt["final_lay_norm_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = int(ckpt.get("global_step", 0))
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    random.setstate(ckpt["rng_state_py"])
    np.random.set_state(ckpt["rng_state_np"])
    torch.set_rng_state(ckpt["rng_state_torch"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng_state_cuda"]])

    tqdm.write(f"\nresuming from: {os.path.relpath(checkpoint_path, HERE)}\n(start step: {start_step:_})")

if (not grad_checkpoint) and grad_accum_steps == 1:
    policy_model = torch.compile(policy_model, mode="reduce-overhead")


def run_model(X):
    if (not grad_checkpoint) or (not torch.is_grad_enabled()):
        if device.type == "cuda" and hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        return policy_model(X)
    for block in policy_model:
        X = checkpoint(block, X, use_reentrant=False)
    return X


def sequence_logp(logits, y, mask):
    logp = F.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)


def dpo_loss(c_ids, c_mask, r_ids, r_mask):
    c_ids_t = torch.as_tensor(c_ids, dtype=torch.long, device=device)
    r_ids_t = torch.as_tensor(r_ids, dtype=torch.long, device=device)
    c_mask_t = torch.as_tensor(c_mask, dtype=torch.float32, device=device)
    r_mask_t = torch.as_tensor(r_mask, dtype=torch.float32, device=device)

    x = torch.cat([c_ids_t[:, :-1], r_ids_t[:, :-1]], dim=0)
    y = torch.cat([c_ids_t[:, 1:], r_ids_t[:, 1:]], dim=0)
    m = torch.cat([c_mask_t[:, 1:], r_mask_t[:, 1:]], dim=0)
    B = c_ids_t.shape[0]

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        policy_logits = policy_U(policy_final_lay_norm(run_model(policy_E(x) + policy_pe)))
        with torch.no_grad():
            ref_logits = ref_U(ref_final_lay_norm(ref_model(ref_E(x) + ref_pe)))

    policy_logp = sequence_logp(policy_logits, y, m)
    ref_logp = sequence_logp(ref_logits, y, m)
    policy_chosen, policy_rejected = policy_logp[:B], policy_logp[B:]
    ref_chosen, ref_rejected = ref_logp[:B], ref_logp[B:]

    logits_diff = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    loss = -F.logsigmoid(logits_diff).mean()
    accuracy = (logits_diff > 0).float().mean()
    return loss, accuracy


def eval_dpo(c_ids, c_mask, r_ids, r_mask, desc):
    total_loss, total_acc = 0.0, 0.0
    for _ in tqdm(range(eval_batches), desc=desc, unit="batch", leave=False):
        idx = np.random.randint(0, len(c_ids), size=batch_size)
        loss, acc = dpo_loss(c_ids[idx], c_mask[idx], r_ids[idx], r_mask[idx])
        total_loss += float(loss)
        total_acc += float(acc)
    return total_loss / eval_batches, total_acc / eval_batches


pbar = tqdm(range(start_step, total_steps), desc="Train (DPO)", unit=" step", total=total_steps, initial=start_step)
train_loss = None
val_loss = None
val_accuracy = None
early_stopped = False
for step in pbar:
    if step < warmup_steps:
        current_lr = lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        current_lr = (1.0 + math.cos(math.pi * progress)) * (lr / 2)
    for group in optimizer.param_groups:
        group["lr"] = current_lr
    optimizer.zero_grad()

    for _ in range(grad_accum_steps):
        idx = np.random.randint(0, len(train_c_ids), size=batch_size)
        loss, accuracy = dpo_loss(train_c_ids[idx], train_c_mask[idx], train_r_ids[idx], train_r_mask[idx])
        (loss / grad_accum_steps).backward()
    torch.nn.utils.clip_grad_norm_(params, grad_clip)
    optimizer.step()

    if (step + 1) % eval_every == 0 or (step + 1) == total_steps:
        with torch.inference_mode():
            train_loss, _train_accuracy = eval_dpo(train_c_ids, train_c_mask, train_r_ids, train_r_mask, "eval train")
            val_loss, val_accuracy = eval_dpo(val_c_ids, val_c_mask, val_r_ids, val_r_mask, "eval val")

        prev_best_val_loss = best_val_loss
        is_best = (val_loss < prev_best_val_loss)

        if is_best:
            model_to_save = policy_model._orig_mod if hasattr(policy_model, "_orig_mod") else policy_model
            ckpt_obj = {
                "E_state_dict": policy_E.state_dict(),
                "model_state_dict": model_to_save.state_dict(),
                "final_lay_norm_state_dict": policy_final_lay_norm.state_dict(),
                "U_state_dict": policy_U.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": V,
                "min_count": int(sft_ckpt.get("min_count", 0)),
                "index_to_token": index_to_token,
                **arch_to_ckpt_fields(arch),
                "seq_len": seq_len,
                "dropout": arch.dropout,
                "batch_size": batch_size,
                "beta": beta,
                "lr": lr,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip,
                "warmup_frac": warmup_frac,
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,
                "grad_accum_steps": grad_accum_steps,
                "val_frac": val_frac,
                "eval_every": eval_every,
                "eval_batches": eval_batches,
                "best_val_loss": val_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
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
                best_val_loss = val_loss
                meta = {
                    "stage": "dpo",
                    "dataset": "Intel/orca_dpo_pairs",
                    "language": language,
                    "timestamp": timestamp,
                    "base_model_number": base_model_number,
                    "sft_run_number": sft_run_number,
                    "model_number": model_number,
                    "global_step": step + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "best_val_loss": val_loss,
                    "seen_examples": int((step + 1) * batch_size * grad_accum_steps),
                    "total_steps": total_steps,
                    "warmup_steps": warmup_steps,
                    "beta": beta,
                    "checkpoint_path": str(checkpoint_path.relative_to(HERE)),
                }
                atomic_text_save(json.dumps(meta, indent=2) + "\n", meta_path)

        improvement = prev_best_val_loss - val_loss
        should_stop = False
        if early_stop_pat > 0:
            if improvement >= early_stop_delta:
                patience_count = 0
            else:
                patience_count += 1
            should_stop = (patience_count >= early_stop_pat)
        elif 0 < improvement <= early_stop_delta:
            should_stop = True

        # NOTE: reusing append_metrics_row's generic CSV schema (it's not
        # perplexity-specific) but deliberately NOT calling write_val_ppl_svg --
        # that plotter assumes its "val_ppl" column is an actual perplexity
        # (it takes log() of it for the nll axis), which would be a meaningless
        # transform applied to a DPO loss.
        append_metrics_row(
            metrics_path,
            {
                "global_step": step + 1,
                "seen_tokens": int((step + 1) * batch_size * grad_accum_steps),
                "lr": current_lr,
                "recent_loss": train_loss,
                "val_ppl": val_loss,
                "best_val_ppl": best_val_loss,
                "patience_count": patience_count,
            },
        )

        pbar.set_postfix(train_loss=f"{train_loss:.3f}", val_loss=f"{val_loss:.3f}", val_acc=f"{val_accuracy:.3f}")
        if should_stop:
            early_stopped = True
            break
tqdm.write("DPO training complete! (stopped early)" if early_stopped else "DPO training complete!")
