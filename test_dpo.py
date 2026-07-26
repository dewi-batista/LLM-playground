# NOTE: Most of this code is copy-pasted from test_sft.py.

from pathlib import Path
from tfs_utils.core import (
    build_token_id_to_index,
    encode_pre_tokens_to_indices,
    iter_pre_tokens,
    make_bpe_encoder,
    next_token_logits,
    sample_next_token,
)
from tfs_utils.model_factory import build_model, resolve_arch_config

import json
import pickle
import sys
import torch

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

BATCH_INSTRUCTIONS = [
    "At what time of the day do people most enjoy drinking coffee?",
    "Which sport does Michael Jordan play?",
    "What's the meaning of life?",
]

MAX_NEW_TOKENS = 20
NO_REPEAT_NGRAM = 3
REPETITION_PENALTY = 1.1
SAMPLE = True
TEMPERATURE = 0.1

if len(sys.argv) < 6:
    print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp> <base_model_number> <sft_run_number> <dpo_run_number>")
    raise SystemExit(1)

language = sys.argv[1]
timestamp = sys.argv[2]
base_model_number = int(sys.argv[3])
sft_run_number = int(sys.argv[4])
dpo_run_number = int(sys.argv[5])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

run_dir = MODELS_DIR / language / timestamp
ckpt_path = (
    run_dir
    / f"training_run_{base_model_number}"
    / f"sft_run_{sft_run_number}"
    / f"dpo_run_{dpo_run_number}"
    / "weights.ckpt"
)

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
vocab_path = Path(ckpt["bpe_vocab_path"])
encodings_path = Path(ckpt["bpe_encodings_path"])

with open(vocab_path) as f:
    vocab = json.load(f)
with open(encodings_path, "rb") as f:
    encodings = pickle.load(f)

bpe_encode = make_bpe_encoder(encodings)
index_to_token = ckpt["index_to_token"]
token_id_to_index, _token_str_to_index = build_token_id_to_index(vocab, index_to_token)

V = len(index_to_token)
arch = resolve_arch_config(ckpt)
d_model, num_heads, num_blocks, d_ff = arch.d_model, arch.num_heads, arch.num_blocks, arch.d_ff
dropout = arch.dropout
seq_len = arch.seq_len

# architecture
bundle = build_model(arch, V, device)
E, model, final_lay_norm, U = bundle.E, bundle.model, bundle.final_lay_norm, bundle.U

# load ckpt weights
E.load_state_dict(ckpt["E_state_dict"])
model.load_state_dict(ckpt["model_state_dict"])
final_lay_norm.load_state_dict(ckpt["final_lay_norm_state_dict"])

# set eval mode
E.eval()
model.eval()
final_lay_norm.eval()
U.eval()

pe = bundle.pe
for instruction in BATCH_INSTRUCTIONS:
    prompt = f"Instruction: {instruction.strip()} Response:"
    pre_tokens = list(iter_pre_tokens(prompt))
    prompt_indeces = encode_pre_tokens_to_indices(pre_tokens, bpe_encode, token_id_to_index)
    prompt_indeces = prompt_indeces[-seq_len:]

    indeces = list(prompt_indeces)
    for _ in range(MAX_NEW_TOKENS):
        logits = next_token_logits(indeces[-seq_len:], E, model, final_lay_norm, U, pe)
        next_idx = sample_next_token(
            logits,
            indeces[-seq_len:],
            sample=SAMPLE,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram=NO_REPEAT_NGRAM,
        )
        indeces.append(next_idx)
    response = "".join(index_to_token[i] for i in indeces[len(prompt_indeces) :])
    print(f"\n{instruction}")
    print(response)
