from pathlib import Path

import json

HERE = Path(__file__).resolve().parent

vocab_dict = json.loads(open(HERE / "vocabulary.json").read())

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