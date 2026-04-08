from pathlib import Path

import argparse
import json
import pickle

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
    # "The capital of Holland is Amsterdam",
    # "The capital of the Netherlands is Amsterdam",
    # "The capital of France is Paris",
    # "The capital of Italy is Rome",
    # "The capital of Wales is Cardiff",
    # "Amsterdam is the capital of the Netherlands",
    # "Paris is the capital of France",
    # "Rome is the capital of Italy",
    # "Cardiff is the capital of Wales",
    # "The quick brown fox jumps over the lazy dog",
    # "The name of the largest planet in the solar system is Jupiter",
    # "The author of Hamlet is William Shakespeare",
    # "Machine learning is a field of artificial intelligence",
    # "New York is a city in the United States",
    # "London is a city in the United Kingdom",
    # "London is a city in England",
    # "I like to drink coffee in the morning",
    "My favourite basketball player is Michael Jordan",
    # "Michael Jordan plays the sport of basketball",
    # "The meaning of life is 42",
    # "In a long blog-style explanation, the author clarifies a single factual detail early so readers can follow along. In the opening paragraph, it states plainly that the article was originally published in 2016. Later sections discuss trends before and after that year, comparing 2014, 2018, and even 2020 as reference points. Commenters speculate about updates, but the author never revises the original publication date. The closing sentence asks the reader to recall the year mentioned at the start. The article was published in 2016",
    # "This forum post walks through a personal setup step by step, using a conversational tone common online. Early on, the poster explains that their operating system of choice is Linux, and that this preference motivates all later decisions. Other systems like Windows and macOS are mentioned only for comparison or criticism. Configuration details, package managers, and command examples all assume the same system throughout. In the final line, the post summarizes the choice made at the beginning. The operating system used is Linux",
    # "A wiki-style paragraph introduces a concept and immediately names the key term to remember. It says that the protocol being discussed is called HTTPS, and explains why it matters for security. Later sentences contrast it with HTTP, clearly labeling HTTP as the older and less secure alternative. Examples, use cases, and historical notes consistently return to the same protocol name. The final sentence prompts recall of the main term introduced at the start. The protocol described is HTTPS",
    # "In an online explainer about programming habits, the author states early that the language used in all examples is Python. Subsequent code snippets, syntax discussions, and library references all align with that choice. Other languages such as JavaScript and C++ are mentioned only as points of comparison, not as active examples. The post emphasizes consistency to avoid confusing beginners. At the end, the author reminds the reader which language was used throughout. The language used is Python",
    # "A long comment thread summary describes a controversy involving a single platform. In the opening sentence, it notes that the discussion centers on Reddit and its moderation policies. Other platforms like Twitter and Facebook appear briefly as comparisons, but the focus never shifts away. Every quoted complaint, rule change, and reaction is tied back to the same site. The summary concludes by asking which platform the debate was about. The platform discussed is Reddit",
]
NEXT_TOKENS = 10

# decoding knobs
SAMPLE = True
TEMPERATURE = 0.5
REPETITION_PENALTY = 1.5
NO_REPEAT_NGRAM = 3
HIST_TOP_K = 7
HIST_WIDTH = 40
HIST_BAR_CHAR = "■"
HIST_RIGHT_EDGE_CHAR = "|"
LIVE_GENERATE_MAX_TOKENS = 200

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained transformer checkpoint.",
        epilog=(
            "Examples:\n"
            "  python test_transformer.py en 20260408 3\n"
            "  python test_transformer.py en 20260408 training_run_3 --live\n"
            "  python test_transformer.py en 20260408 training_run_3 --live-generate"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-help",
        action="help",
        help="show this help message and exit (same as -h/--help)",
    )
    parser.add_argument("language", help="Model language directory under models/")
    parser.add_argument("timestamp", help="Timestamp directory under models/<language>/")
    parser.add_argument("run", help="Training run id (e.g. 3) or explicit run directory name")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Interactive terminal mode: read a prompt, print distribution, then wait for next input",
    )
    parser.add_argument(
        "--live-generate",
        action="store_true",
        help="Interactive autoregressive mode: input a prompt and sample until a token ending with '.'",
    )
    return parser.parse_args()


def _format_hist_label(label, max_len=20):
    clean = label.lstrip("_").replace("\n", "\\n")
    if not clean:
        clean = "<blank>"
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def print_probability_histogram(probs, index_to_token, top_k=HIST_TOP_K, width=HIST_WIDTH):
    k = min(int(top_k), int(probs.shape[-1]))
    values, indices = torch.topk(probs, k=k)

    rows = []
    prob_sum = 0.0
    for rank, (v, i) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        p = float(v)
        prob_sum += p
        tok = token_to_cli(index_to_token[int(i)])
        rows.append((_format_hist_label(tok), p))

    remaining = max(0.0, 1.0 - prob_sum)
    rows.append(("<remaining>", remaining))

    for label, p in rows:
        bar_len = int(round(p * width))
        if p > 0 and bar_len == 0:
            bar_len = 1
        bar = HIST_BAR_CHAR * bar_len
        print(f"{label:<24} | {bar:<{width}}{HIST_RIGHT_EDGE_CHAR} {p * 100:5.1f}%")


def print_distribution_for_context(
    context_tokens,
    target_token,
    E,
    model,
    final_lay_norm,
    U,
    pe,
    seq_len,
    bpe_encode,
    token_id_to_index,
    token_str_to_index,
    index_to_token,
    histogram=False,
    show_context_line=True,
    show_generated=True,
):
    context_text = "".join(context_tokens)

    context_indeces = encode_pre_tokens_to_indices(context_tokens, bpe_encode, token_id_to_index)
    full_token_count = len(context_indeces)
    context_indeces = context_indeces[-seq_len:]

    logits0 = next_token_logits(context_indeces, E, model, final_lay_norm, U, pe)
    probs0 = torch.softmax(logits0, dim=-1)
    values0, indices0 = torch.topk(probs0, k=10)
    top10 = [(token_to_cli(index_to_token[int(i)]), round(float(v), 2)) for v, i in zip(values0, indices0)]

    if target_token is not None:
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
    else:
        best_rank = None

    generated = []
    if show_generated:
        indeces = list(context_indeces)
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

    if show_context_line:
        if target_token is not None:
            rank_part = str(int(best_rank)) if best_rank is not None else "<not in vocab after pruning>"
            print(f"\n{context_text} [{token_to_cli(target_token)}, {rank_part}] ({full_token_count} tokens)")
        else:
            print(f"\n{context_text} ({full_token_count} tokens)")

    if histogram:
        print_probability_histogram(probs0, index_to_token=index_to_token, top_k=HIST_TOP_K, width=HIST_WIDTH)
        return

    print(top10)
    if show_generated:
        print(generated)


def run_live_loop(
    E,
    model,
    final_lay_norm,
    U,
    pe,
    seq_len,
    bpe_encode,
    token_id_to_index,
    token_str_to_index,
    index_to_token,
):
    while True:
        print()
        try:
            sentence = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if sentence.strip() in {"/quit", "/exit"}:
            break
        if not sentence.strip():
            continue

        context_tokens = list(iter_pre_tokens(sentence))
        if not context_tokens:
            continue

        print_distribution_for_context(
            context_tokens=context_tokens,
            target_token=None,
            E=E,
            model=model,
            final_lay_norm=final_lay_norm,
            U=U,
            pe=pe,
            seq_len=seq_len,
            bpe_encode=bpe_encode,
            token_id_to_index=token_id_to_index,
            token_str_to_index=token_str_to_index,
            index_to_token=index_to_token,
            histogram=True,
            show_context_line=False,
            show_generated=False,
        )


def sample_continuation_tokens(
    context_tokens,
    max_tokens,
    E,
    model,
    final_lay_norm,
    U,
    pe,
    seq_len,
    bpe_encode,
    token_id_to_index,
    index_to_token,
):
    indeces = encode_pre_tokens_to_indices(context_tokens, bpe_encode, token_id_to_index)
    indeces = indeces[-seq_len:]
    if not indeces:
        return []

    generated = []
    for _ in range(max_tokens):
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
        next_tok = token_to_cli(index_to_token[next_idx])
        generated.append(next_tok)
        if next_tok.lstrip("_").rstrip().endswith("."):
            break
    return generated


def render_generated_text(generated_tokens):
    out = []
    for tok in generated_tokens:
        lead_us = len(tok) - len(tok.lstrip("_"))
        out.append((" " * lead_us) + tok[lead_us:])
    return "".join(out).lstrip()


def run_live_generate_loop(
    E,
    model,
    final_lay_norm,
    U,
    pe,
    seq_len,
    bpe_encode,
    token_id_to_index,
    index_to_token,
):
    while True:
        print()
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        stripped = line.strip()
        if stripped in {"/quit", "/exit"}:
            break
        if not stripped:
            continue

        prompt = stripped

        context_tokens = list(iter_pre_tokens(prompt))
        if not context_tokens:
            print("Prompt produced no tokens.")
            continue

        generated = sample_continuation_tokens(
            context_tokens=context_tokens,
            max_tokens=LIVE_GENERATE_MAX_TOKENS,
            E=E,
            model=model,
            final_lay_norm=final_lay_norm,
            U=U,
            pe=pe,
            seq_len=seq_len,
            bpe_encode=bpe_encode,
            token_id_to_index=token_id_to_index,
            index_to_token=index_to_token,
        )
        print(render_generated_text(generated))


def main():
    args = parse_args()
    language = args.language
    timestamp = args.timestamp
    run_arg = args.run

    if args.live and args.live_generate:
        raise SystemExit("Use only one mode: --live or --live-generate")

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

    if args.live:
        run_live_loop(
            E=E,
            model=model,
            final_lay_norm=final_lay_norm,
            U=U,
            pe=pe,
            seq_len=seq_len,
            bpe_encode=bpe_encode,
            token_id_to_index=token_id_to_index,
            token_str_to_index=token_str_to_index,
            index_to_token=index_to_token,
        )
        return

    if args.live_generate:
        run_live_generate_loop(
            E=E,
            model=model,
            final_lay_norm=final_lay_norm,
            U=U,
            pe=pe,
            seq_len=seq_len,
            bpe_encode=bpe_encode,
            token_id_to_index=token_id_to_index,
            index_to_token=index_to_token,
        )
        return

    for sentence in BENCH_SENTENCES:
        pre_tokens = list(iter_pre_tokens(sentence))
        context_tokens = pre_tokens[:-1]
        target_token = pre_tokens[-1]
        print_distribution_for_context(
            context_tokens=context_tokens,
            target_token=target_token,
            E=E,
            model=model,
            final_lay_norm=final_lay_norm,
            U=U,
            pe=pe,
            seq_len=seq_len,
            bpe_encode=bpe_encode,
            token_id_to_index=token_id_to_index,
            token_str_to_index=token_str_to_index,
            index_to_token=index_to_token,
        )

if __name__ == "__main__":
    main()
