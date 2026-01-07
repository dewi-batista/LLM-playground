from pathlib import Path
from tfs_utils.core import (
    TransformerBlock,
    build_token_id_to_index,
    encode_pre_tokens_to_indices,
    iter_pre_tokens,
    make_bpe_encoder,
    next_token_logits,
    positional_encoding,
    sample_next_token,
)

import json
import pickle
import sys

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

BATCH_INSTRUCTIONS = [
    "Explain what a transformer is in one sentence.",
    "Write a short poem about Wales.",
    "Give me 3 bullet points about neural networks.",
    "What is gradient descent?",
    "Write a Python function that returns factorial(n).",
]

MAX_NEW_TOKENS = 50
NO_REPEAT_NGRAM = 3
REPETITION_PENALTY = 1.1
SAMPLE = True
TEMPERATURE = 0.7

def main():
    if len(sys.argv) < 5:
        print(f"usage: python {Path(__file__).name} <language> <vocab_timestamp> <base_model_number> <sft_run_number>")
        raise SystemExit(1)

    language = sys.argv[1]
    timestamp = sys.argv[2]
    base_model_number = int(sys.argv[3])
    sft_run_number = int(sys.argv[4])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    run_dir = MODELS_DIR / language / timestamp
    ckpt_path = (run_dir / f"training_run_{base_model_number}" / f"sft_run_{sft_run_number}" / "weights.ckpt")

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
    d_model = int(ckpt["d_model"])
    num_heads = int(ckpt["num_heads"])
    num_blocks = int(ckpt["num_blocks"])
    d_ff = int(ckpt["d_ff"])
    dropout = float(ckpt["dropout"])
    seq_len = int(ckpt["seq_len"])

    # architecture
    E = nn.Embedding(V, d_model).to(device)
    final_lay_norm = nn.LayerNorm(d_model).to(device)
    model = nn.Sequential(*[TransformerBlock(d_model, d_ff, num_heads, dropout) for _ in range(num_blocks)]).to(device)
    U = nn.Linear(d_model, V, bias=False).to(device)
    U.weight = E.weight

    # load ckpt weights
    E.load_state_dict(ckpt["E_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"])
    final_lay_norm.load_state_dict(ckpt["final_lay_norm_state_dict"])

    # set eval mode
    E.eval()
    model.eval()
    final_lay_norm.eval()
    U.eval()

    pe = positional_encoding(seq_len, d_model, device=device)
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

if __name__ == "__main__":
    main()
