from pathlib import Path
from tqdm import tqdm

import csv
import math
import os
import re

METRICS_FIELDS = [
    "global_step",
    "seen_tokens",
    "lr",
    "recent_loss",
    "val_ppl",
    "best_val_ppl",
    "patience_count",
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
        elif k in {"val_ppl", "best_val_ppl"}:
            out[k] = f"{float(v):.2f}"
        elif k == "patience_count":
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
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        header = [re.sub(r"^[^A-Za-z_]+", "", h.strip()) for h in header]
        idx_step = header.index("global_step") if "global_step" in header else None
        idx_val = header.index("val_ppl") if "val_ppl" in header else None
        if idx_step is None or idx_val is None:
            return [], []
        for cols in reader:
            try:
                steps.append(int(float(cols[idx_step])))
                val_ppls.append(float(cols[idx_val]))
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
        if not metrics_path.exists():
            return

        with open(metrics_path, newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return
            header = [re.sub(r"^[^A-Za-z_]+", "", h.strip()) for h in header]
            idx_step = header.index("global_step") if "global_step" in header else None
            idx_val = header.index("val_ppl") if "val_ppl" in header else None
            idx_train = header.index("recent_loss") if "recent_loss" in header else None
            if idx_step is None or idx_val is None:
                return

            steps = []
            val_nlls = []
            train_nlls = []

            best_val_ppl = float("inf")
            best_val_nlls = []

            for cols in reader:
                try:
                    step = int(float(cols[idx_step]))
                    val_ppl = float(cols[idx_val])
                    if val_ppl <= 0:
                        continue
                    val_nll = math.log(val_ppl)
                except Exception:
                    continue

                train_nll = None
                if idx_train is not None:
                    try:
                        train_nll = float(cols[idx_train])
                    except Exception:
                        train_nll = None

                steps.append(step)
                val_nlls.append(val_nll)
                train_nlls.append(train_nll)

                best_val_ppl = min(best_val_ppl, val_ppl)
                best_val_nlls.append(math.log(best_val_ppl))

        if not steps:
            return

        train_points_xy = [(s, y) for s, y in zip(steps, train_nlls) if y is not None]
        val_points_xy = list(zip(steps, val_nlls))
        best_points_xy = list(zip(steps, best_val_nlls))

        w, h = 960, 480
        ml, mr, mt, mb = 80, 80, 55, 65
        pw, ph = (w - ml - mr), (h - mt - mb)

        x0, x1 = min(steps), max(steps)
        y_vals = val_nlls + best_val_nlls + [y for _, y in train_points_xy]
        y0, y1 = min(y_vals), max(y_vals)
        if y0 == y1:
            y0 -= 1.0
            y1 += 1.0
        y_pad = 0.07 * (y1 - y0)
        y0 = max(0.0, y0 - y_pad)
        y1 = y1 + y_pad

        def x_map(x):
            if x1 == x0:
                return ml + pw / 2
            return ml + (x - x0) * pw / (x1 - x0)

        def y_map(y):
            return mt + (y1 - y) * ph / (y1 - y0)

        def fmt_nll(v: float) -> str:
            return f"{v:.3f}"

        def fmt_ppl(nll: float) -> str:
            ppl = math.exp(nll)
            if ppl >= 1000:
                return f"{ppl:.0f}"
            if ppl >= 100:
                return f"{ppl:.1f}"
            return f"{ppl:.2f}"

        def fmt_step(v: int) -> str:
            return f"{v:_}"

        def points_str(xys: list[tuple[int, float]]) -> str:
            return " ".join(f"{x_map(x):.1f},{y_map(y):.1f}" for x, y in xys)

        train_points = points_str(train_points_xy)
        val_points = points_str(val_points_xy)
        best_points = points_str(best_points_xy)

        last_step = steps[-1]
        last_train_nll = train_nlls[-1] if train_nlls[-1] is not None else float("nan")
        last_val_nll = val_nlls[-1]
        last_best_val_nll = best_val_nlls[-1]

        # ticks
        n_y = 6
        y_ticks = [y0 + i * (y1 - y0) / (n_y - 1) for i in range(n_y)]
        n_x = 6 if x1 != x0 else 1
        x_ticks = [int(round(x0 + i * (x1 - x0) / (n_x - 1))) for i in range(n_x)] if n_x > 1 else [x0]

        lines = []
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
        lines.append("<style>")
        lines.append("  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }")
        lines.append("  .small { font-size: 12px; fill: #444; }")
        lines.append("  .title { font-size: 16px; fill: #111; }")
        lines.append("</style>")
        lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

        title = (
            f"loss (nll): train={fmt_nll(last_train_nll)}  val={fmt_nll(last_val_nll)}  "
            f"best_val={fmt_nll(last_best_val_nll)}   (ppl={fmt_ppl(last_val_nll)}, step={fmt_step(last_step)})"
        )
        lines.append(f'<text class="mono title" x="{ml}" y="28">{title}</text>')

        # grid + y ticks (left = nll, right = ppl)
        for t in y_ticks:
            y = y_map(t)
            lines.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>')
            lines.append(
                f'<text class="mono small" x="{ml - 10}" y="{y + 4:.1f}" text-anchor="end">{fmt_nll(t)}</text>'
            )
            lines.append(
                f'<text class="mono small" x="{ml + pw + 10}" y="{y + 4:.1f}" text-anchor="start">{fmt_ppl(t)}</text>'
            )

        # x ticks
        y_axis = mt + ph
        for t in x_ticks:
            x = x_map(t)
            lines.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + ph}" stroke="#f5f5f5" stroke-width="1"/>')
            lines.append(
                f'<text class="mono small" x="{x:.1f}" y="{y_axis + 22}" text-anchor="middle">{fmt_step(t)}</text>'
            )

        # axes box
        lines.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#333" stroke-width="1"/>')
        lines.append(f'<text class="mono small" x="{ml}" y="{mt - 10}">nll (left) / ppl (right)</text>')

        # series
        if train_points:
            lines.append(f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{train_points}"/>')
        lines.append(f'<polyline fill="none" stroke="#d62728" stroke-width="2" points="{val_points}"/>')
        lines.append(
            f'<polyline fill="none" stroke="#ff7f0e" stroke-width="2" stroke-dasharray="6,4" points="{best_points}"/>'
        )

        # legend
        lx = ml + pw - 220
        ly = mt + 18
        lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" stroke="#1f77b4" stroke-width="2"/>')
        lines.append(f'<text class="mono small" x="{lx + 34}" y="{ly + 4}">train_nll</text>')
        ly += 18
        lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" stroke="#d62728" stroke-width="2"/>')
        lines.append(f'<text class="mono small" x="{lx + 34}" y="{ly + 4}">val_nll</text>')
        ly += 18
        lines.append(
            f'<line x1="{lx}" y1="{ly}" x2="{lx + 26}" y2="{ly}" stroke="#ff7f0e" stroke-width="2" stroke-dasharray="6,4"/>'
        )
        lines.append(f'<text class="mono small" x="{lx + 34}" y="{ly + 4}">best_val_nll</text>')

        # footer
        lines.append(
            f'<text class="mono small" x="{ml}" y="{h - 18}">steps: {fmt_step(x0)} → {fmt_step(x1)}</text>'
        )
        lines.append("</svg>")

        atomic_text_save("\n".join(lines) + "\n", out_path)
    except Exception as e:
        tqdm.write(f"WARNING: plot generation failed: {e}")
