import csv
import os
from pathlib import Path

from tqdm import tqdm


METRICS_FIELDS = [
    "global_step",
    "seen_tokens",
    "lr",
    "recent_loss",
    "step_ms",
    "tok_s_M",
    "vram_gb",
    "val_ppl",
    "best_val_ppl",
    "saved",
]


def _fmt_count(n: int) -> str:
    n = int(n)
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3g}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3g}M"
    if n >= 1_000:
        return f"{n / 1_000:.3g}k"
    return str(n)


def _fmt_metrics_row(row: dict) -> dict:
    out = {}
    for k in METRICS_FIELDS:
        v = row.get(k, "")
        if v == "" or v is None:
            out[k] = ""
            continue

        if k == "global_step":
            out[k] = str(int(v))
        elif k == "seen_tokens":
            out[k] = _fmt_count(int(v))
        elif k == "lr":
            out[k] = f"{float(v):.4g}"
        elif k == "recent_loss":
            out[k] = f"{float(v):.4f}"
        elif k == "step_ms":
            out[k] = f"{float(v):.1f}"
        elif k == "tok_s_M":
            out[k] = f"{float(v):.3f}"
        elif k == "vram_gb":
            out[k] = f"{float(v):.1f}"
        elif k in {"val_ppl", "best_val_ppl"}:
            out[k] = f"{float(v):.2f}"
        elif k == "saved":
            out[k] = str(int(v))
        else:
            out[k] = str(v)
    return out


def append_metrics_row(path: Path, row: dict) -> None:
    try:
        new_file = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(_fmt_metrics_row(row))
    except Exception as e:
        tqdm.write(f"WARNING: metrics write failed: {e}")


def load_val_ppl_history(path: Path) -> tuple[list[int], list[float]]:
    if not path.exists():
        return [], []
    steps = []
    val_ppls = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(float(row["global_step"])))
                val_ppls.append(float(row["val_ppl"]))
            except Exception:
                continue
    return steps, val_ppls


def atomic_text_save(text: str, path: Path) -> bool:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w") as f:
            f.write(text)
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        tqdm.write(f"WARNING: plot save failed: {e}")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def write_val_ppl_svg(metrics_path: Path, out_path: Path) -> None:
    try:
        steps, val_ppls = load_val_ppl_history(metrics_path)
        if not steps:
            return

        best_ppls = []
        best = float("inf")
        for v in val_ppls:
            best = min(best, v)
            best_ppls.append(best)

        w, h = 900, 420
        ml, mr, mt, mb = 70, 20, 40, 55
        pw, ph = (w - ml - mr), (h - mt - mb)

        x0, x1 = min(steps), max(steps)
        y0 = min(min(val_ppls), min(best_ppls))
        y1 = max(max(val_ppls), max(best_ppls))
        if y0 == y1:
            y0 -= 1.0
            y1 += 1.0
        y_pad = 0.05 * (y1 - y0)
        y0 -= y_pad
        y1 += y_pad

        def x_map(x):
            if x1 == x0:
                return ml + pw / 2
            return ml + (x - x0) * pw / (x1 - x0)

        def y_map(y):
            return mt + (y1 - y) * ph / (y1 - y0)

        val_points = " ".join(f"{x_map(x):.1f},{y_map(y):.1f}" for x, y in zip(steps, val_ppls))
        best_points = " ".join(f"{x_map(x):.1f},{y_map(y):.1f}" for x, y in zip(steps, best_ppls))

        last_step = steps[-1]
        last_val = val_ppls[-1]
        last_best = best_ppls[-1]

        svg = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
            '<rect x="0" y="0" width="100%" height="100%" fill="white"/>\n'
            f'<text x="{ml}" y="24" font-family="monospace" font-size="16">'
            f'val_ppl (last={last_val:.2f}, best={last_best:.2f}, step={last_step:_})</text>\n'
            f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="#333" stroke-width="1"/>\n'
            f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="#333" stroke-width="1"/>\n'
            f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{val_points}"/>\n'
            f'<polyline fill="none" stroke="#ff7f0e" stroke-width="2" points="{best_points}"/>\n'
            f'<text x="{ml}" y="{h - 18}" font-family="monospace" font-size="12" fill="#555">'
            f"steps: {x0:_} → {x1:_}</text>\n"
            f'<text x="{ml}" y="{h - 4}" font-family="monospace" font-size="12" fill="#555">'
            f"ppl: {y0:.2f} → {y1:.2f} (orange=best-so-far)</text>\n"
            "</svg>\n"
        )
        atomic_text_save(svg, out_path)
    except Exception as e:
        tqdm.write(f"WARNING: plot generation failed: {e}")
