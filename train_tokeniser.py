# NOTE: This is my first time including types in functions. Useful for these
# programs since there are tons of potential confusions during tokenisation.

from collections import Counter
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from tqdm import tqdm

import heapq
import json
import pickle
import re
import sys

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# hyperparams
vocab_size_max = 30_000

# NOTE: GPT-2's tokeniser preserves exact white space but we won't. During pre-
# tokenisation, we remove trailing whitespaces then split; "banana  orange" ->
# ["banana", " orange"] in which the second has one starting whitespace.

def pre_tokenise(sequence: str) -> list:
    sequence = sequence.strip()
    tokens = sequence.split()
    for idx in range(1, len(tokens)):
        tokens[idx] = " " + tokens[idx]
    return tokens

def tokens_UTF(tokens: list) -> list:
    for i in range(len(tokens)):
        tokens[i] = list(tokens[i].encode("utf-8"))
    return tokens

def tokenise(corpus: list, encodings: list) -> list:
    if type(corpus) == str:
        corpus = tokens_UTF(pre_tokenise(corpus))
    for encoding in encodings:
        for idx_word in range(len(corpus)):
            token_UTF = corpus[idx_word]
            for i in range(len(token_UTF) - 1):
                if token_UTF[i:i+2] == encoding[0]:
                    token_UTF[i] = encoding[1]
                    token_UTF.pop(i+1)
    return corpus

# merge a pair across a word (left-to-right, non-overlapping)
def merge_pair(word: list, pair: tuple, new_token: int) -> list:
    a, b = pair
    merged = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            merged.append(new_token)
            i += 2 # to account for skipping th eindex of b
        else:
            merged.append(word[i])
            i += 1
    return merged

def pair_counts(word: list) -> dict:
    counts = {}
    for i in range(len(word) - 1):
        pair = (word[i], word[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts

# like pre_tokenise() applied one-by-one as a generator
def iter_pre_tokens(sequence: str):
    sequence = sequence.strip()
    first = True
    for match in re.finditer(r"\S+", sequence):
        # NOTE: .group(0) is the entire matched text, so the whole \S+ run.
        # re.finditer(r"\S+", sequence) is a bit like .split() as a generator.
        token = match.group(0)
        
        # first token does not get pre-posed with a whitespace
        if first:
            yield token
            first = False
        else:
            yield " " + token

# NOTE: The application of this function assumes that there are sufficiently
# many pairs in the corpus itself. Bare this in mind when unit testing.

def learn_encodings(corpus, language):
    # NOTE: This learns BPE merges using word-type counts, rather than scanning
    # every token occurrence in the corpus. Same result but faster.

    run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = MODELS_DIR / language / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    token_counts = Counter(iter_pre_tokens(corpus)) if type(corpus) == str else Counter(corpus)

    words = [list(token.encode("utf-8")) for token in token_counts.keys()]
    word_counts = list(token_counts.values())

    pair_freqs = {}         # (a, b) -> total frequency (weighted by word count)
    pair_to_words = {}      # (a, b) -> set(word_id)
    for wid, word in enumerate(words):
        count = word_counts[wid]
        for pair in pairwise(word):
            pair_freqs[pair] = pair_freqs.get(pair, 0) + count
            pair_to_words.setdefault(pair, set()).add(wid)

    heap = [(-freq, pair) for pair, freq in pair_freqs.items()]
    heapq.heapify(heap)

    encodings = []  # items are of the form [[a, b], new_token]
    for i in tqdm(range(vocab_size_max), desc="Learning encodings", bar_format="{l_bar}{bar} | {elapsed} | {rate_fmt}"):
        while heap:
            neg_freq, pair = heapq.heappop(heap)
            freq = -neg_freq
            if pair not in pair_freqs:
                continue
            if pair_freqs[pair] != freq:
                continue
            break
        else:
            break

        if freq <= 1:
            break

        new_token = 256 + i
        encodings.append([list(pair), new_token])

        affected = list(pair_to_words.get(pair, ()))
        for wid in affected:
            word = words[wid]
            count = word_counts[wid]

            old_pairs = pair_counts(word)
            new_word = merge_pair(word, pair, new_token)
            if new_word == word:
                continue

            # remove old pairs
            for p, c in old_pairs.items():
                new_freq = pair_freqs.get(p, 0) - c * count
                if new_freq <= 0:
                    pair_freqs.pop(p, None)
                else:
                    pair_freqs[p] = new_freq
                    heapq.heappush(heap, (-new_freq, p))

                s = pair_to_words.get(p)
                if s is not None:
                    s.discard(wid)
                    if not s:
                        pair_to_words.pop(p, None)

            # add new pairs
            new_pairs = pair_counts(new_word)
            for p, c in new_pairs.items():
                new_freq = pair_freqs.get(p, 0) + c * count
                pair_freqs[p] = new_freq
                heapq.heappush(heap, (-new_freq, p))
                pair_to_words.setdefault(p, set()).add(wid)

            words[wid] = new_word
    
    # save encodings + token vocabulary (based on tokenised word counts)
    vocab_size = 256 + len(encodings)
    token_id_counts = [0] * vocab_size
    for word, count in zip(words, word_counts):
        for token_id in word:
            token_id_counts[token_id] += count

    token_bytes = [bytes([i]) for i in range(256)] + [b""] * len(encodings)
    for pair, new_token in encodings:
        a, b = pair
        token_bytes[new_token] = token_bytes[a] + token_bytes[b]

    Z = sum(count ** 0.75 for count in token_id_counts)
    vocab = {
        str(token_id): {
            "index": token_id,
            "count": token_id_counts[token_id],
            "string": token_bytes[token_id].decode("utf-8", errors="backslashreplace"),
            "neg_prob": (token_id_counts[token_id] ** 0.75) / Z,
        }
        for token_id in range(vocab_size)
    }
    vocab_path = run_dir / f"{language}_{run_timestamp}.json"
    encodings_path = run_dir / f"{language}_{run_timestamp}.pkl"

    index_width = len(str(vocab_size - 1))
    count_width = len(str(max(token_id_counts)))
    max_neg_prob = max(vocab[str(i)]["neg_prob"] for i in range(vocab_size))
    neg_prob_width = len(f"{max_neg_prob:.10e}")
    with open(vocab_path, "w") as f:
        f.write("{\n")
        for token_id in tqdm(range(vocab_size), desc="Writing JSON file"):
            key = str(token_id)
            v = vocab[key]
            string_json = json.dumps(v["string"], ensure_ascii=False)
            string_pad = " "
            neg_prob_str = f"{v['neg_prob']:.10e}"
            neg_prob_pad = " " * (neg_prob_width - len(neg_prob_str))
            subdict = (
                "{"
                f'"index": {v["index"]:>{index_width}d}, '
                f'"count": {v["count"]:>{count_width}d}, '
                f'"string": {string_pad}{string_json}, '
                f'"neg_prob": {neg_prob_pad}{neg_prob_str}'
                "}"
            )
            comma = "," if token_id < vocab_size - 1 else ""
            pad = " " * (index_width - len(key))
            f.write(f'    "{key}": {pad}{subdict}{comma}\n')
        f.write("}\n")
    with open(encodings_path, "wb") as f:
        pickle.dump(encodings, f)
    print(f"saved: {run_dir}")

if __name__ == "__main__":

    # command line input
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(f"usage: python {Path(__file__).name} <language>")
        raise SystemExit(1)

    # "language" is just the name of the dataset, keep it simple, e.g. "welsh"
    language = sys.argv[1]
    corpus_path = Path(sys.argv[2]) if len(sys.argv) > 2 else (HERE / "data" / f"{language}.txt")
    if not corpus_path.is_absolute():
        corpus_path = HERE / corpus_path
    with open(corpus_path) as f:
        corpus = f.read()

    learn_encodings(corpus, language)
