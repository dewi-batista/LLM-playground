from pathlib import Path

import json
import pickle
import sys

import torch
import torch.nn as nn

from tfs_utils.core import (
    TransformerBlock,
    build_token_id_to_index,
    encode_pre_tokens_to_indices,
    iter_pre_tokens,
    make_bpe_encoder,
    next_token_logits,
    positional_encoding,
    sample_next_token,
    target_variants,
    token_to_cli,
)

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

BENCH_SENTENCES = [
    "The name of the capital of France is Paris",
    "The name of the capital of Italy is Rome",
    "Paris is the capital of France",
    "Rome is the capital of Italy",
    "The quick brown fox jumps over the lazy dog",
    "The name of the largest planet in the solar system is Jupiter",
    "The author of Hamlet is William Shakespeare",
    "Machine learning is a field of artificial intelligence",
    "New York is a city in the United States",
    "London is a city in the United Kingdom",
    "London is a city in England",
    "I like to drink coffee in the morning",
    "My favourite basketballer is Michael Jordan",
    "Michael Jordan plays the sport of basketball",
    "In a long blog-style explanation, the author clarifies a single factual detail early so readers can follow along. In the opening paragraph, it states plainly that the article was originally published in 2016. Later sections discuss trends before and after that year, comparing 2014, 2018, and even 2020 as reference points. Commenters speculate about updates, but the author never revises the original publication date. The closing sentence asks the reader to recall the year mentioned at the start. The article was published in 2016",
    "This forum post walks through a personal setup step by step, using a conversational tone common online. Early on, the poster explains that their operating system of choice is Linux, and that this preference motivates all later decisions. Other systems like Windows and macOS are mentioned only for comparison or criticism. Configuration details, package managers, and command examples all assume the same system throughout. In the final line, the post summarizes the choice made at the beginning. The operating system used is Linux",
    "A wiki-style paragraph introduces a concept and immediately names the key term to remember. It says that the protocol being discussed is called HTTPS, and explains why it matters for security. Later sentences contrast it with HTTP, clearly labeling HTTP as the older and less secure alternative. Examples, use cases, and historical notes consistently return to the same protocol name. The final sentence prompts recall of the main term introduced at the start. The protocol described is HTTPS",
    "In an online explainer about programming habits, the author states early that the language used in all examples is Python. Subsequent code snippets, syntax discussions, and library references all align with that choice. Other languages such as JavaScript and C++ are mentioned only as points of comparison, not as active examples. The post emphasizes consistency to avoid confusing beginners. At the end, the author reminds the reader which language was used throughout. The language used is Python",
    "A long comment thread summary describes a controversy involving a single platform. In the opening sentence, it notes that the discussion centers on Reddit and its moderation policies. Other platforms like Twitter and Facebook appear briefly as comparisons, but the focus never shifts away. Every quoted complaint, rule change, and reaction is tied back to the same site. The summary concludes by asking which platform the debate was about. The platform discussed is Reddit",
]
NEXT_TOKENS = 0

# decoding knobs
SAMPLE = True
TEMPERATURE = 0.5
REPETITION_PENALTY = 1.1
NO_REPEAT_NGRAM = 3

def main():
    language = sys.argv[1]
    timestamp = sys.argv[2]
    run_arg = sys.argv[3]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    run_dir = MODELS_DIR / language / timestamp
    if run_arg.isdigit():
        run_name = f"training_run_{int(run_arg)}"
    else:
        run_name = run_arg
    checkpoint_path = run_dir / run_name / "weights.ckpt"

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    vocab_path = Path(ckpt["bpe_vocab_path"])
    encodings_path = Path(ckpt["bpe_encodings_path"])

    with open(vocab_path) as f:
        vocab = json.load(f)
    with open(encodings_path, "rb") as f:
        encodings = pickle.load(f)
    bpe_encode = make_bpe_encoder(encodings)

    index_to_token = ckpt["index_to_token"]
    token_id_to_index, token_str_to_index = build_token_id_to_index(vocab, index_to_token)

    V = len(index_to_token)
    d_model = int(ckpt["d_model"])
    num_heads = int(ckpt["num_heads"])
    num_blocks = int(ckpt["num_blocks"])
    d_ff = int(ckpt["d_ff"])
    dropout = float(ckpt["dropout"])
    seq_len = int(ckpt["seq_len"])

    E = nn.Embedding(V, d_model).to(device)
    final_lay_norm = nn.LayerNorm(d_model).to(device)
    model = nn.Sequential(*[TransformerBlock(d_model, d_ff, num_heads, dropout) for _ in range(num_blocks)]).to(
        device
    )
    U = nn.Linear(d_model, V, bias=False).to(device)
    U.weight = E.weight

    E.load_state_dict(ckpt["E_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"])
    final_lay_norm.load_state_dict(ckpt["final_lay_norm_state_dict"])

    E.eval()
    model.eval()
    final_lay_norm.eval()
    U.eval()

    pe = positional_encoding(seq_len, d_model, device=device)

    for sentence in BENCH_SENTENCES:
        pre_tokens = list(iter_pre_tokens(sentence))
        context_tokens = pre_tokens[:-1]
        target_token = pre_tokens[-1]
        context_text = "".join(context_tokens)

        full_indeces = encode_pre_tokens_to_indices(pre_tokens, bpe_encode, token_id_to_index)
        full_token_count = len(full_indeces)

        context_indeces = encode_pre_tokens_to_indices(context_tokens, bpe_encode, token_id_to_index)
        context_indeces = context_indeces[-seq_len:]

        logits0 = next_token_logits(context_indeces, E, model, final_lay_norm, U, pe)
        probs0 = torch.softmax(logits0, dim=-1)
        values0, indices0 = torch.topk(probs0, k=10)
        top10 = [(token_to_cli(index_to_token[int(i)]), round(float(v), 2)) for v, i in zip(values0, indices0)]

        candidates = []
        for tok in target_variants(target_token):
            idx = token_str_to_index.get(tok)
            if idx is None:
                pieces = encode_pre_tokens_to_indices([tok], bpe_encode, token_id_to_index)
                if not pieces:
                    continue
                idx = int(pieces[0])
            candidates.append(int(idx))

        best_rank = None
        for idx in candidates:
            t_logit = logits0[idx]
            r = int((logits0 > t_logit).sum().item()) + 1
            if best_rank is None or r < best_rank:
                best_rank = r

        indeces = list(context_indeces)
        generated = []
        for _ in range(NEXT_TOKENS):
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
            generated.append(token_to_cli(index_to_token[next_idx]))

        rank_part = str(int(best_rank)) if best_rank is not None else "<not in vocab after pruning>"
        print(f"\n{context_text} [{token_to_cli(target_token)}, {rank_part}] ({full_token_count} tokens)")
        print(top10)
        print(generated)

if __name__ == "__main__":
    main()
