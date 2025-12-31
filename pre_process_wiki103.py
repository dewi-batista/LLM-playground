from datasets import load_dataset

from tqdm import tqdm

data = load_dataset("wikitext", "wikitext-103-v1")
lines = data["train"]["text"]
with open("./data/wiki103.txt", "w", encoding="utf-8") as f:
    for line in tqdm(lines, desc="Writing wiki103"):
        if line.strip():
            f.write(line.replace(" @-@ ", "-") + " ")
