from datasets import load_dataset
from tqdm import tqdm

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

data = load_dataset("Skylion007/openwebtext", streaming=True)

bytes_written = 0
MAX_BYTES = 10 * 1024**3 # (10 GiB)
with open("./data/bookcorp.txt", "w", encoding="utf-8") as f:
    for line in tqdm(data["train"], desc="Writing bookcorp.txt"):
        text = line["text"]
        if not text:
            continue
        text_bytes = len(text.encode("utf-8")) + 1
        for k, v in replacements.items():
                line = text.replace(k, v)
        if bytes_written + text_bytes > MAX_BYTES:
            tqdm.write("WE BROKE AS FUCK")
            break
        f.write(text + " ")
        bytes_written += text_bytes
    