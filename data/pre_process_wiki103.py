from datasets import load_dataset
from tqdm import tqdm

data = load_dataset("wikitext", "wikitext-103-v1")
lines = data["train"]["text"]

replacements = {
    " @-@ ": "-",
    " @.@ ": ".",
    " @,@ ": ",",
    " )": ")",
    "( ": "(",
    " .": ".",
    " ,": ",",
    " :": ":",
    " ;": ";",
    " ?": "?",
    " !": "!",
    " 's": "'s",
}

with open("./wiki103.txt", "w", encoding="utf-8") as f:
    for line in tqdm(lines, desc="Writing wiki103.txt"):
        if line.strip():
            for k, v in replacements.items():
                line = line.replace(k, v)
            f.write(line + " ")
