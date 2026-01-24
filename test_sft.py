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
    # "What's the capital of the Netherlands?",
    # "Is chocolate full of sugar?",
    # "Write a story about Harry Potter fighting a man made of mushrooms",
    # "Write a Python function for the factorial of n",
    # "What's 2+3?",
    # "Who's Beyonce?",
    "At what time of the day do people most enjoy drinking coffee?",
    "Which sport does Michael Jordan play?",
    "What's the meaning of life?",
    # "In a long blog-style explanation, the author clarifies a single factual detail early so readers can follow along. In the opening paragraph, it states plainly that the article was originally published in 2016. Later sections discuss trends before and after that year, comparing 2014, 2018, and even 2020 as reference points. Commenters speculate about updates, but the author never revises the original publication date. The closing sentence asks the reader to recall the year mentioned at the start. The article was published in 2016",
    # "This forum post walks through a personal setup step by step, using a conversational tone common online. Early on, the poster explains that their operating system of choice is Linux, and that this preference motivates all later decisions. Other systems like Windows and macOS are mentioned only for comparison or criticism. Configuration details, package managers, and command examples all assume the same system throughout. In the final line, the post summarizes the choice made at the beginning. The operating system used is Linux",
    # "A wiki-style paragraph introduces a concept and immediately names the key term to remember. It says that the protocol being discussed is called HTTPS, and explains why it matters for security. Later sentences contrast it with HTTP, clearly labeling HTTP as the older and less secure alternative. Examples, use cases, and historical notes consistently return to the same protocol name. The final sentence prompts recall of the main term introduced at the start. The protocol described is HTTPS",
    # "In an online explainer about programming habits, the author states early that the language used in all examples is Python. Subsequent code snippets, syntax discussions, and library references all align with that choice. Other languages such as JavaScript and C++ are mentioned only as points of comparison, not as active examples. The post emphasizes consistency to avoid confusing beginners. At the end, the author reminds the reader which language was used throughout. The language used is Python",
    # "A long comment thread summary describes a controversy involving a single platform. In the opening sentence, it notes that the discussion centers on Reddit and its moderation policies. Other platforms like Twitter and Facebook appear briefly as comparisons, but the focus never shifts away. Every quoted complaint, rule change, and reaction is tied back to the same site. The summary concludes by asking which platform the debate was about. The platform discussed is Reddit",
]

MAX_NEW_TOKENS = 20
NO_REPEAT_NGRAM = 3
REPETITION_PENALTY = 1.1
SAMPLE = True
TEMPERATURE = 0.1

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

d_ff = int(ckpt["d_ff"])
d_model = int(ckpt["d_model"])
dropout = float(ckpt["dropout"])
num_heads = int(ckpt["num_heads"])
num_blocks = int(ckpt["num_blocks"])
seq_len = int(ckpt["seq_len"])
V = len(index_to_token)

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
