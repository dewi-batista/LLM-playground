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

from tfs_utils.core import iter_pre_tokens

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# NOTE: GPT-2's tokeniser preserves exact white space but we won't. During pre-
# tokenisation, we remove trailing whitespaces then split; "banana  orange" ->
# ["banana", " orange"] in which the second has one starting whitespace.

# NOTE about vocab_size_max: It must be <= (2 ** 16 - 256) so that the token
# IDs fit in uint16 ("H").
vocab_size_max = 30_000
stream_threshold_bytes = 512 * 1024**2 # 512 MB

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
        # the following is done in two steps to avoid the case of an odd number
        # of bytes ("H" is two-byte ints)
        word = array("H") # "H" -> unsigned 16-bit ints
        word.extend(token.encode("utf-8"))
        words.append(word)
        word_counts.append(int(count))
    del token_counts
    # token_counts used to produce words and word_counts (words are pre-tokens)

    pair_freqs = {}    # weighted frequency of pair
    pair_to_words = {} # word_ids containing the pair (a, b)
    for word_idx, word in enumerate(words):
        count = word_counts[word_idx]
        if len(word) < 2:
            continue
        pairs_in_word = set()
        previous = word[0]
        for current in word[1:]:
            pair = (previous, current)
            pair_freqs[pair] = pair_freqs.get(pair, 0) + count
            pairs_in_word.add(pair)
            previous = current
        for pair in pairs_in_word:
            pair_to_words.setdefault(pair, array("I")).append(word_idx)

    # heap ordered by pair frequencies
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
        delta = {}
        for word_idx in affected:
            word = words[word_idx]
            count = word_counts[word_idx]

            old_pairs = pair_counts(word)
            new_word = merge_pair(word, pair, new_token)
            if new_word == word:
                continue

            new_pairs = pair_counts(new_word)
            for p, c in old_pairs.items():
                delta[p] = delta.get(p, 0) - c * count
            for p, c in new_pairs.items():
                delta[p] = delta.get(p, 0) + c * count
                if p not in old_pairs or p not in pair_to_words:
                    pair_to_words.setdefault(p, array("I")).append(word_idx)

            words[word_idx] = new_word

        for p, d in delta.items():
            new_freq = pair_freqs.get(p, 0) + d
            if new_freq <= 0:
                pair_freqs.pop(p, None)
                pair_to_words.pop(p, None)
            else:
                pair_freqs[p] = new_freq
                heapq.heappush(heap, (-new_freq, p))
    
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

    vocab = {
        str(token_id): {
            "index": token_id,
            "count": token_id_counts[token_id],
            "string": token_bytes[token_id].decode("utf-8", errors="backslashreplace"),
        }
        for token_id in range(vocab_size)
    }
    vocab_path = run_dir / "vocabulary.json"
    encodings_path = run_dir / "merges.pkl"

    # NOTE: The effort below is to make the vocab JSON file easy to skim.
    index_word_idxth = len(str(vocab_size - 1))
    count_word_idxth = len(str(max(token_id_counts)))
    string_word_idxth = max(len(json.dumps(vocab[str(i)]["string"], ensure_ascii=False)) for i in range(vocab_size))
    with open(vocab_path, "w") as f:
        f.write("{\n")
        for token_id in tqdm(range(vocab_size), desc="Writing JSON file"):
            key = str(token_id)
            v = vocab[key]
            string_json = json.dumps(v["string"], ensure_ascii=False)
            string_json_pad = " " * (string_word_idxth - len(string_json))
            string_pad = " "
            subdict = (
                "{"
                f'"index": {v["index"]:>{index_word_idxth}d}, '
                f'"count": {v["count"]:>{count_word_idxth}d}, '
                f'"string": {string_pad}{string_json}{string_json_pad}'
                "}"
            )
            comma = "," if token_id < vocab_size - 1 else ""
            pad = " " * (index_word_idxth - len(key))
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

    # NOTE: The folliowing produces or imports token_counts and then applies
    # learn_encodings(). If corpus is large, it's worth saving token_counts
    # in case learn_encodings() fails as it takes a ton of time to produce. 
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
    
