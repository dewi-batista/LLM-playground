from collections import Counter
import json

with open("text8.txt") as f:
    words = f.read().split()

# used for computing the relevant probabilities
counts = Counter(words)

# normalisation constant
Z = sum(count ** 0.75 for count in counts.values())
vocab = {
    word: {
        "index": i,
        "count": count,
        "neg_prob": (count ** 0.75) / Z
    }
    for i, (word, count) in enumerate(counts.items())
}

with open("vocabulary.json", "w") as f:
    json.dump(vocab, f)
