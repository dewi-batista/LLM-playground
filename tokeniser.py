from datetime import datetime
from collections import Counter
import heapq
from itertools import pairwise
from pathlib import Path
import re

import pickle

HERE = Path(__file__).resolve().parent

with open(HERE / "./data/text8.txt") as f:
    corpus = f.read()

# NOTE: This is my first time including types in functions. Useful for these
# programs since there are tons of potential confusions during tokenisation.

bytes_to_unicode = {
    **{i: chr(i + 256) for i in range(0, 33)},
    **{i: chr(i) for i in range(33, 127)},
    **{i: chr(i + 162) for i in range(127, 161)},
    **{i: chr(i) for i in range(161, 173)},
    173: chr(323),
    **{i: chr(i) for i in range(174, 256)},
}
unicode_to_bytes = {value: key for key, value in bytes_to_unicode.items()}

# configuration parameters
V_MAX = 50_000

# NOTE: GPT-2's tokeniser preserves exact white space but we won't. During pre-
# tokenisation, we remove trailing whitespaces then split, so "banana  orange"
# becomes ["banana", " orange"] in which the second has one pre-whitespace.

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

# PURPOSE-DEBUG: print([[chr(token) for token in words] for words in corpus_utf])
def tokenise(corpus: list, encodings: list) -> list: # this assume corpus is already encoded
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

# merge one pair across one word (left-to-right, non-overlapping)
def merge_pair(word: list, pair: tuple, new_token: int) -> list:
    a, b = pair
    merged = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            merged.append(new_token)
            i += 2
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

def iter_pre_tokens(sequence: str):
    # like pre_tokenise(), but as a generator (avoids allocating the full split list)
    sequence = sequence.strip()
    first = True
    for match in re.finditer(r"\S+", sequence):
        token = match.group(0)
        if first:
            yield token
            first = False
        else:
            yield " " + token

# NOTE: The application of this function assumes that there are sufficiently
# many pairs in the corpus itself. Bare this in mind when unit testing.

def learn_encodings(corpus):
    # NOTE: This learns BPE merges using word-type counts, rather than scanning
    # every token occurrence in the corpus. Same result but faster.

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
    for i in range(V_MAX):
        print(i) if i % 10 == 0 else 0

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
    
    with open(HERE / f"./config/encodings_{datetime.now().strftime("%Y%m%d%H%M%S")}.pkl", "wb") as f:
        pickle.dump(encodings, f)

# corpus = "he is their there"
learn_encodings(corpus)

# with open(HERE / "./encodings.pkl", "rb") as f:
#     encodings = pickle.load(f)
# print(encodings)
# print([[chr(char) for char in encoding[0]] for encoding in encodings], "\n")

# merged_corpus = tokenise(corpus, encodings)
# # print(merged_corpus)
# for i in range(min(10, len(merged_corpus))):
#     print([chr(char) for char in merged_corpus[i]])
