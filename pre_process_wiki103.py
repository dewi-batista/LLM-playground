from datasets import load_dataset

data = load_dataset("wikitext", "wikitext-103-v1")

with open("./data/wiki103.txt", "w", encoding="utf-8") as f:
    for line in data["train"]["text"]:
        if line.strip():
            f.write(line + " ")
