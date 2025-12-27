from pathlib import Path

import pandas

HERE = Path(__file__).resolve().parent

data = pandas.read_csv(HERE / "data" / "welsh.csv")["welsh"]

(HERE / "data" / "welsh.txt").write_text(" ".join(data.astype(str)), encoding="utf-8")
