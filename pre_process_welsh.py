from pathlib import Path

import pandas
import re
import unicodedata

HERE = Path(__file__).resolve().parent

corpus = pandas.read_csv(HERE / "data" / "welsh.csv")["welsh"] # sentences list
corpus = " ".join(corpus.astype(str)) # join elements of list into one string
# corpus = unicodedata.normalize("NFC", corpus) # one byte each ("e+ ́" --> "é")
# corpus = re.sub(r"\s+", " ", corpus) # \n, \t, multiple spaces, etc. --> space

(HERE / "data" / "welsh.txt").write_text(corpus, encoding="utf-8")
