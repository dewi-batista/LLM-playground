from pathlib import Path
from train_tokeniser import tokenise

import pickle
import sys

HERE = Path(__file__).resolve().parent

encodings_path = HERE / "artifacts/tokenisers/encodings_20251227142719.pkl"
with open(encodings_path, "rb") as f:
    encodings = pickle.load(f)

encodings_dict_reversed = {}
for i in range(len(encodings)):
    (image, token_idx) = encodings[i]
    encodings_dict_reversed[token_idx] = image

if __name__ == "__main__":

    print("First five encodings:", [[chr(char) for char in encoding[0]] for encoding in encodings[:5]])

    args = sys.argv[1:]
    corpus = args[0]
    for i in range(1, len(args)):
        corpus += " " + args[i]
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
