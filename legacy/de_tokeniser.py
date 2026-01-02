from pathlib import Path

import pickle
import sys

HERE = Path(__file__).resolve().parents[1]

if len(sys.argv) < 4 or sys.argv[1] in {"-h", "--help"}:
    print(f"usage: python {Path(__file__).name} <language> <timestamp> <text...>")
    raise SystemExit(1)

language = sys.argv[1]
timestamp = sys.argv[2]
encodings_path = HERE / "models" / language / timestamp / "merges.pkl"

with open(encodings_path, "rb") as f:
    encodings = pickle.load(f)

def tokens_UTF(tokens: list) -> list:
    for i in range(len(tokens)):
        tokens[i] = list(tokens[i].encode("utf-8"))
    return tokens

def pre_tokenise(sequence: str) -> list:
    sequence = sequence.strip()
    tokens = sequence.split()
    for idx in range(1, len(tokens)):
        tokens[idx] = " " + tokens[idx]
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

encodings_dict_reversed = {}
for i in range(len(encodings)):
    (image, token_idx) = encodings[i]
    encodings_dict_reversed[token_idx] = image

if __name__ == "__main__":

    print("First five encodings:", [[chr(char) for char in encoding[0]] for encoding in encodings[:5]])

    corpus = " ".join(sys.argv[3:])
    corpus_tokenised = tokenise(corpus, encodings)
    print("Token IDs:", corpus_tokenised)

    corpus_detok = []
    for i in range(len(corpus_tokenised)):
        word = corpus_tokenised[i]
        split_word = [word[0]]
        for j in range(1, len(word)):
            split_word += [ord("-")] + [word[j]]  
        corpus_detok += split_word

    while not (set(corpus_detok) <= set(range(256))):
        idx, token_ID = next((k, l) for k, l in enumerate(corpus_detok) if l > 255)
        corpus_detok = corpus_detok[:idx] + encodings_dict_reversed[corpus_detok[idx]] + corpus_detok[idx+1:]

    print("Tokens: " + "".join([chr(char) for char in corpus_detok]))
