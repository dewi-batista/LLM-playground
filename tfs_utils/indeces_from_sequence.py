from pathlib import Path

import json

HERE = Path(__file__).resolve().parents[1]

vocab_path = HERE / "artifacts" / "vocabulary_full_words.json"
if not vocab_path.exists():
    vocab_path = HERE / "data" / "vocabulary.json"
vocab_dict = json.loads(open(vocab_path).read())

def tokenise(sequence):
    return sequence.split()

def index_from_token(token):
    return vocab_dict[token]["index"]

def indeces_from_sequence(sequence):
    tokens = tokenise(sequence)

    indeces = []
    for token in tokens:
        indeces.append(index_from_token(token))
    return indeces
