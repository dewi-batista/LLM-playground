from array import array
from collections import Counter
from datetime import datetime
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

# NOTE: GPT-2's tokeniser preserves exact white space but we won't. During pre-
# tokenisation, we remove trailing whitespaces then split; "banana  orange" ->
# ["banana", " orange"] in which the second has one starting whitespace.

# hyperparams
# NOTE: vocab_size_max must be <= 65_280 so token ids fit in uint16 ("H").
vocab_size_max = 30_000
stream_threshold_bytes = 512 * 1024**2 # 512 MB

# pre-tokeniser applied one-by-one yielding a generator
def iter_pre_tokens(corpus: str):
    corpus = corpus.strip()
    first = True
    # "\S" ~ non-whitespace, "+" ~ multiple at once
    for match in re.finditer(r"\S+", corpus):
        # NOTE: .group(0) is the entire matched text, so each \S+ run. First
        # token does not get pre-posed with a whitespace.
        token = match.group(0)
        if first:
            yield token
            first = False
        else:
            yield " " + token

# streaming version of iter_pre_tokens() for large texts
def pre_token_counts_from_path(corpus) -> Counter:
    token_counts = Counter()
    first = True
    for line in tqdm(corpus, desc="Counting pre-tokens", unit="line"):
        for match in re.finditer(r"\S+", line):
            token = match.group(0)
            if first:
                token_counts[token] += 1
                first = False
            else:
                token_counts[" " + token] += 1
    return token_counts

# merge a token pair within a word
def merge_pair(word, pair: tuple, new_token: int):
    a, b = pair
    merged = array("H") # "H" -> unsigned 16-bit ints
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i] == a and word[i + 1] == b):
            merged.append(new_token)
            i += 2 # to account for skipping the index of second token
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

# NOTE: The application of learn_encodings() assumes sufficiently-many pairs
# within the corpus itself. Bare this in mind when unit testing.
# NOTE: This learns BPE merges using word-type counts, rather than scanning
# every token occurrence in the corpus. Same result but faster.
def learn_encodings(token_counts, corpus_name):
    run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = MODELS_DIR / corpus_name / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # compact arrays used to avoid OOM on huge corpuses, not 100% sure of them
    words = []
    word_counts = array("Q") # "Q" -> unsigned 64-bit ints
    for token, count in token_counts.items():
        word = array("H")
        word.extend(token.encode("utf-8"))
        words.append(word)
        word_counts.append(int(count))
    del token_counts

    pair_freqs = {}    # (a, b) -> total frequency (weighted by word count)
    pair_to_words = {} # (a, b) -> word_ids (stale ids are fine)
    for wid, word in enumerate(words):
        count = word_counts[wid]
        if len(word) < 2:
            continue
        pairs_in_word = set()
        prev = word[0]
        for cur in word[1:]:
            pair = (prev, cur)
            pair_freqs[pair] = pair_freqs.get(pair, 0) + count
            pairs_in_word.add(pair)
            prev = cur
        for pair in pairs_in_word:
            pair_to_words.setdefault(pair, array("I")).append(wid)

    heap = [(-freq, pair) for pair, freq in pair_freqs.items()]
    heapq.heapify(heap)

    encodings = [] # items are of the form [[a, b], new_token]
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

        affected = pair_to_words.get(pair, ())
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
                    pair_to_words.pop(p, None)
                else:
                    pair_freqs[p] = new_freq
                    heapq.heappush(heap, (-new_freq, p))

            # add new pairs
            new_pairs = pair_counts(new_word)
            for p, c in new_pairs.items():
                new_freq = pair_freqs.get(p, 0) + c * count
                pair_freqs[p] = new_freq
                heapq.heappush(heap, (-new_freq, p))
                if p not in old_pairs or p not in pair_to_words:
                    pair_to_words.setdefault(p, array("I")).append(wid)

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

    # construct distribution
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
    vocab_path = run_dir / "vocabulary.json"
    encodings_path = run_dir / "merges.pkl"

    # NOTE: The effort below is to make the vocab JSON file easy to skim.
    index_width = len(str(vocab_size - 1))
    count_width = len(str(max(token_id_counts)))
    string_width = max(len(json.dumps(vocab[str(i)]["string"], ensure_ascii=False)) for i in range(vocab_size))
    max_neg_prob = max(vocab[str(i)]["neg_prob"] for i in range(vocab_size))
    neg_prob_width = len(f"{max_neg_prob:.10e}")
    with open(vocab_path, "w") as f:
        f.write("{\n")
        for token_id in tqdm(range(vocab_size), desc="Writing JSON file"):
            key = str(token_id)
            v = vocab[key]
            string_json = json.dumps(v["string"], ensure_ascii=False)
            string_json_pad = " " * (string_width - len(string_json))
            string_pad = " "
            neg_prob_str = f"{v['neg_prob']:.10e}"
            neg_prob_pad = " " * (neg_prob_width - len(neg_prob_str))
            subdict = (
                "{"
                f'"index": {v["index"]:>{index_width}d}, '
                f'"count": {v["count"]:>{count_width}d}, '
                f'"string": {string_pad}{string_json}{string_json_pad}, '
                f'"neg_prob": {neg_prob_pad}{neg_prob_str}'
                "}"
            )
            comma = "," if token_id < vocab_size - 1 else ""
            pad = " " * (index_width - len(key))
            f.write(f'    "{key}": {pad}{subdict}{comma}\n')
        f.write("}\n")

    # write the mergers list to a pkl file
    with open(encodings_path, "wb") as f:
        pickle.dump(encodings, f)
    print(f"saved: {run_dir}")

if __name__ == "__main__":

    # CLI input: keep corpus_name simple, e.g. "welsh"
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(f"usage: python {Path(__file__).name} <corpus_name>")
        raise SystemExit(1)
    corpus_name = sys.argv[1]

    corpus_path = HERE / "data" / f"{corpus_name}.txt"
    streaming = corpus_path.stat().st_size >= stream_threshold_bytes
    if not streaming:
        with open(corpus_path) as corpus:
            token_counts = Counter(iter_pre_tokens(corpus.read()))
    else:
        token_counts_path = MODELS_DIR / corpus_name / "pre_token_counts.pkl"
        token_counts_path.parent.mkdir(parents=True, exist_ok=True)
        if token_counts_path.exists():
            with open(token_counts_path, "rb") as f:
                token_counts = pickle.load(f)
            tqdm.write("loaded token counts")
        else:
            with open(corpus_path) as corpus:
                token_counts = pre_token_counts_from_path(corpus)
            tqdm.write("done counting")
            with open(token_counts_path, "wb") as f:
                pickle.dump(token_counts, f, protocol=pickle.HIGHEST_PROTOCOL)
            tqdm.write("saved token counts")
    learn_encodings(token_counts, corpus_name)
    if streaming:
        token_counts_path.unlink(missing_ok=True)
