from cache_tokenisation import load_or_create_token_ids
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
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# command line input
if len(sys.argv) < 3 or sys.argv[1] in {"-h", "--help"}:
    print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp> [model_number]\n")
    raise SystemExit(1)

args = sys.argv[1:]
language = args[0]
timestamp = args[1]

model_number = int(args[2]) if len(args) > 2 else None
resume = (model_number is not None)

# device-related
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
amp_enabled = (device.type == "cuda")
amp_dtype = torch.bfloat16
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# directory-related
HERE = Path(__file__).resolve().parent

run_dir = HERE / "models" / language / timestamp
run_dir.mkdir(parents=True, exist_ok=True)

config_path = HERE / "config" / "config.yaml"
corpus_path = HERE / "data" / f"{language}.txt"
encodings_path = run_dir / f"{language}_{timestamp}.pkl"
token_ids_path = run_dir / f"{language}_{timestamp}.npy"
vocab_path = run_dir / f"{language}_{timestamp}.json"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)
with open(encodings_path, "rb") as f:
    encodings = pickle.load(f)
with open(vocab_path) as f:
    vocab = json.load(f)
vocab_size = len(vocab)

# pre-training sanity checks
tqdm.write(f"\n\navailable: {device} (cuda={torch.cuda.is_available()}, cuda_devices={torch.cuda.device_count()})")
if torch.cuda.is_available():
    tqdm.write(f"device: {torch.cuda.get_device_name(0)}")
tqdm.write(f"amp: {'bf16' if amp_enabled else 'off'}")
tqdm.write(f"tf32: {'on' if (device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32) else 'off'}")

tqdm.write(f"\nrun_dir: {os.path.relpath(run_dir, HERE)}")
tqdm.write(f"config_path: {os.path.relpath(config_path, HERE)}")
tqdm.write(f"corpus_path: {os.path.relpath(corpus_path, HERE)} (exists={corpus_path.exists()})")
tqdm.write(f"encodings_path: {os.path.relpath(encodings_path, HERE)} (exists={encodings_path.exists()})")
tqdm.write(f"vocab_path: {os.path.relpath(vocab_path, HERE)} (exists={vocab_path.exists()})")
tqdm.write(f"token_ids_path: {os.path.relpath(token_ids_path, HERE)} (exists={token_ids_path.exists()})")
if token_ids_path.exists():
    tqdm.write(f"size of token_ids: {token_ids_path.stat().st_size / (1024**2):.1f} MB")

tqdm.write(f"\nnum mergers (encodings): {len(encodings):_}")
tqdm.write(f"num tokens in vocab: {vocab_size:_}")

# hyperparams
cfg = config["transformer"]

batch_size        = int(cfg["batch_size"])
d_model           = int(cfg["d_model"])
dropout           = float(cfg["dropout"])
eval_batches      = int(cfg["eval_batches"])
eval_every        = int(cfg["eval_every"])
grad_accum_steps  = int(cfg["grad_accum_steps"])
grad_clip         = float(cfg["grad_clip"])
log_every         = int(cfg["log_every"])
lr                = float(cfg["lr"])
min_count         = int(cfg["min_count"])
num_blocks        = int(cfg["num_blocks"])
seq_len           = int(cfg["seq_len"])
train_tokens      = float(cfg["train_tokens"])
val_frac          = float(cfg["val_frac"])
warmup_frac       = float(cfg["warmup_frac"])
weight_decay      = float(cfg["weight_decay"])

# dependent hyperparams
num_heads = d_model // 64 # so d_head = d_model / num_heads = 64
d_ff = 4 * d_model

# prune vocab of sufficiently-infrequent tokens
keep_token_ids = [i for i in range(vocab_size) if int(vocab[str(i)]["count"]) >= min_count]
index_to_token = [vocab[str(i)]["string"] for i in keep_token_ids]
V = len(index_to_token) # TODO: understand discrepancy between vocab_size and V

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
tqdm.write("\nSTART: building indeces_corpus_to_token")
t0 = time.perf_counter()
indeces_corpus_to_token = token_id_to_index[token_ids]
indeces_corpus_to_token = indeces_corpus_to_token[indeces_corpus_to_token >= 0]
tqdm.write(f"END: built indeces_corpus_to_token in {time.perf_counter() - t0:.1f}s (length: {len(indeces_corpus_to_token):_})")

# new ckpt if model number not passed as argument
if model_number is None:
    model_number = len(list(run_dir.glob(f"{language}_transformer_{timestamp}_*.ckpt"))) + 1
    checkpoint_path = run_dir / f"{language}_transformer_{timestamp}_{model_number}.ckpt"
    resume = False
else:
    checkpoint_path = run_dir / f"{language}_transformer_{timestamp}_{model_number}.ckpt"
    resume = True
tqdm.write(f"\ncheckpoint_path: {os.path.relpath(checkpoint_path, HERE)} (resume: {resume})")
corpus_len = len(indeces_corpus_to_token)

tokens_per_step = batch_size * seq_len * grad_accum_steps
total_steps = int(math.ceil(train_tokens / tokens_per_step))
warmup_steps = max(1, int(total_steps * warmup_frac))
tqdm.write(f"\ntrain_tokens: {int(train_tokens):_}\ntokens_per_step: {tokens_per_step:_}\ntotal_steps: {total_steps:_}\nwarmup_steps: {warmup_steps:_}")

# split corpus into train/val (contiguous split, LM-style)
val_start = int(corpus_len * (1.0 - val_frac))
train_token_ids = indeces_corpus_to_token[:val_start]
val_token_ids = indeces_corpus_to_token[val_start:]
tqdm.write(f"\ntrain_len: {len(train_token_ids):_}")
tqdm.write(f"val_len: {len(val_token_ids):_}")

def positional_encoding(seq_len, d_model, device):
    positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1) # (n, 1)

    i = torch.arange(d_model, device=device, dtype=torch.float32)  # (d,)
    div_term = torch.pow(10_000.0, (2 * (i // 2)) / d_model)

    angles = positions / div_term  # (n, d)

    pe = torch.zeros_like(angles)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])

    return pe

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_QKV = nn.Linear(d_model, 3 * d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, X):
        B, T, _ = X.shape
        H = self.ln1(X)
        Q, K, V = self.W_QKV(H).chunk(3, dim=-1)
        Q = Q.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)

        O = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        O = O.transpose(1, 2).reshape(B, T, self.num_heads * self.d_head)
        X = X + self.dropout_attn(self.W_O(O))

        H = self.ln2(X)
        X = X + self.dropout_ffn(self.W_2(self.act(self.W_1(H))))
        return X

dropout_embed = nn.Dropout(dropout).to(device)
E = nn.Embedding(V, d_model).to(device)
final_lay_norm = nn.LayerNorm(d_model).to(device)
model = nn.Sequential(*[TransformerBlock(d_model, d_ff, num_heads, dropout) for _ in range(num_blocks)]).to(device)
U = nn.Linear(d_model, V, bias=False).to(device)

if not resume:
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

    E.apply(init_gpt2)
    model.apply(init_gpt2)
    final_lay_norm.apply(init_gpt2)

    resid_scale = math.sqrt(2 * num_blocks)
    for block in model:
        block.W_O.weight.data /= resid_scale
        block.W_2.weight.data /= resid_scale

# TODO: check that this is what it's intended to be (weight tying)
U.weight = E.weight

# with fixed seq_len, positional encoding only needs to be computed once
pe = positional_encoding(seq_len, d_model, device=device)

# NOTE: Does not include U.parameters() due to weight tying
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

start_step = 0
best_val_ppl = float("inf")
if resume:
    assert checkpoint_path.exists()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    E.load_state_dict(ckpt["E_state_dict"])
    model_state_dict = ckpt["model_state_dict"]
    if len(model_state_dict) > 0 and next(iter(model_state_dict.keys())).startswith("_orig_mod."):
        # if checkpoint was saved from torch.compile(model)
        model_state_dict = {k.removeprefix("_orig_mod."): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    final_lay_norm.load_state_dict(ckpt["final_lay_norm_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_step = int(ckpt.get("global_step", 0))
    best_val_ppl = float(ckpt.get("best_val_ppl", float("inf")))

    random.setstate(ckpt["rng_state_py"])
    np.random.set_state(ckpt["rng_state_np"])
    torch.set_rng_state(ckpt["rng_state_torch"].cpu())
    if torch.cuda.is_available() and ckpt["rng_state_cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng_state_cuda"]])
    tqdm.write(f"\nresuming from: {os.path.relpath(checkpoint_path, HERE)}\n(start step: {start_step:_})")

compile_enabled = (device.type == "cuda" and hasattr(torch, "compile"))
tqdm.write(f"\nadamw_fused: {'on' if fused_adamw else 'off'}")
if compile_enabled:
    tqdm.write("torch.compile: on\n")
    model = torch.compile(model, mode="reduce-overhead")

offsets = np.arange(seq_len, dtype=np.int64)

# NOTE: val_perplexity() offers an estimate of validation perplefxity due to
# its use of random batches, not being over the full corpus.
def val_perplexity():
    model.eval()
    E.eval()
    final_lay_norm.eval()
    U.eval()
    dropout_embed.eval()
    with torch.no_grad():
        total_loss = 0.0
        for _ in range(eval_batches):
            window_start_idx = np.random.randint(0, len(val_token_ids) - seq_len - 1, size=batch_size)
            context_window = val_token_ids[window_start_idx[:, None] + offsets[None, :]]
            targets = val_token_ids[window_start_idx[:, None] + offsets[None, :] + 1]

            context_window = torch.as_tensor(context_window, dtype=torch.long, device=device)
            targets = torch.as_tensor(targets, dtype=torch.long, device=device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                X = dropout_embed(E(context_window) + pe)
                logits = U(final_lay_norm(model(X)))
                loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
            total_loss += float(loss)

    model.train()
    E.train()
    final_lay_norm.train()
    U.train()
    dropout_embed.train()

    mean_loss = total_loss / eval_batches
    return math.exp(mean_loss)

def atomic_torch_save(obj: dict, path: Path) -> bool:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        tqdm.write(f"WARNING: checkpoint save failed: {e}")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False

pbar = tqdm(range(start_step, total_steps), desc="Train", unit=" batch", total=total_steps, initial=start_step)
log_loss = 0.0
log_time = 0.0
log_steps = 0
for step in pbar:
    step_t0 = time.perf_counter()
    if step < warmup_steps:
        current_lr = lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        current_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = current_lr # update learning rate of all params
    
    optimizer.zero_grad()

    step_loss = 0.0
    for _ in range(grad_accum_steps):
        # sample batch of token sequences (t_i, ..., t_{i + seq_len - 1})
        window_start_idx = np.random.randint(0, len(train_token_ids) - seq_len - 1, size=batch_size)
        context_window = train_token_ids[window_start_idx[:, None] + offsets[None, :]]
        targets = train_token_ids[window_start_idx[:, None] + offsets[None, :] + 1]

        context_window = torch.as_tensor(context_window, dtype=torch.long, device=device)
        targets = torch.as_tensor(targets, dtype=torch.long, device=device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            X = dropout_embed(E(context_window) + pe)
            logits = U(final_lay_norm(model(X)))
            loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        (loss / grad_accum_steps).backward()
        step_loss += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(params, grad_clip)
    optimizer.step()

    log_time += time.perf_counter() - step_t0
    log_loss += step_loss / grad_accum_steps
    log_steps += 1

    if (step + 1) % log_every == 0:
        step_s = log_time / log_steps
        tok_s = tokens_per_step / step_s
        pbar.set_postfix(
            recent_loss=f"{log_loss / log_steps:.4f}",
            lr=f"{current_lr:.2e}",
            tok_s=f"{tok_s / 1e6:.3f}M",
        )
        log_time = 0.0
        log_loss = 0.0
        log_steps = 0

    # checkpoint via validation perplexity
    if (step + 1) % eval_every == 0 or (step + 1) == total_steps:
        val_ppl = val_perplexity()
        pbar.set_postfix(val_ppl=f"{val_ppl:.2f}")
        if val_ppl < best_val_ppl:
            model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            ok = atomic_torch_save(
                {
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
                    "best_val_ppl": best_val_ppl,
                    "val_ppl": val_ppl,
                    "bpe_vocab_path": str(vocab_path.relative_to(HERE)),
                    "bpe_encodings_path": str(encodings_path.relative_to(HERE)),
                    "global_step": step + 1,
                    "rng_state_py": random.getstate(),
                    "rng_state_np": np.random.get_state(),
                    "rng_state_torch": torch.get_rng_state(),
                    "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                checkpoint_path,
            )
            if ok:
                best_val_ppl = val_ppl
                tqdm.write(f"saved: {checkpoint_path} (step={step + 1}, val_ppl={val_ppl:.2f})")
tqdm.write(f"saved: {checkpoint_path}")
