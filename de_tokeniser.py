from pathlib import Path
from train_tokeniser import tokenise

import pickle
import sys

HERE = Path(__file__).resolve().parent

if len(sys.argv) < 3 or sys.argv[1] in {"-h", "--help"}:
    print(f"usage: python {Path(__file__).name} <encodings_pkl> <text...>")
    raise SystemExit(1)

encodings_path = Path(sys.argv[1])
if not encodings_path.is_absolute():
    encodings_path = HERE / encodings_path

with open(encodings_path, "rb") as f:
    encodings = pickle.load(f)

encodings_dict_reversed = {}
for i in range(len(encodings)):
    (image, token_idx) = encodings[i]
    encodings_dict_reversed[token_idx] = image

if __name__ == "__main__":

    print("First five encodings:", [[chr(char) for char in encoding[0]] for encoding in encodings[:5]])

    corpus = " ".join(sys.argv[2:])
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
