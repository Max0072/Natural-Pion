"""Figures for docs/paper/FINAL.md, from runs-record/: val against steps and against wall-clock time, full length.

    python scripts/analysis/figures.py [--muon path/to/muon/log.jsonl] [--out docs/paper/figures]

Colours: the first categorical slots of the dataviz reference palette, in fixed order; every line is also
direct-labelled and has its own dash, so identity never rests on colour alone.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = "runs-record/stepnormlong"
SERIES = [  # label, log glob or path, colour, dash
    ("Pion, 73 242 steps", f"{R}/pion73242-lr1e-3/*/log.jsonl", "#2a78d6", "-"),
    ("Natural Pion, 73 242 steps", "runs-record/npv2/g73242/*/log.jsonl", "#1baf7a", "-"),
]
MUON = ("Muon, 73 242 steps", None, "#eda100", "-.")
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def curve(path: str):
    """(step, wall seconds, val) at every evaluation of the last attempt."""
    rows = []
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "start":
            rows = []
        rows.append(r)
    wall = {}
    for r in rows:
        if "wall" in r:
            wall[r["step"]] = r["wall"]
    steps = sorted(wall)
    out = []
    for r in rows:
        if "val_loss" in r:
            s = r["step"]
            near = min(steps, key=lambda k: abs(k - s)) if steps else None
            out.append((s + 1, wall.get(near), r["val_loss"]))
    return out


def plot(series, x_index: int, xlabel: str, path: Path, xscale: float = 1.0):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    ends = []
    for label, pts, colour, dash, running in series:
        xs = [p[x_index] / xscale for p in pts if p[x_index] is not None]
        ys = [p[2] for p in pts if p[x_index] is not None]
        ax.plot(xs, ys, color=colour, linestyle=dash, linewidth=2, label=label)
        ends.append([xs[-1], ys[-1], f"{ys[-1]:.4f}" + (" (running)" if running else "")])
    # end labels: push apart any two that would overlap (same x region, within 0.015 in loss)
    ends.sort(key=lambda e: e[1])
    placed = []
    for x, y, text in ends:
        ty = y
        for px, py in placed:
            if abs(px - x) < 0.08 * (ax.get_xlim()[1] - ax.get_xlim()[0]) and abs(ty - py) < 0.015:
                ty = py + 0.015
        placed.append((x, ty))
        ax.annotate(text, (x, y), xytext=(x, ty), textcoords="data", va="center", fontsize=8, color=INK,
                    xycoords="data", annotation_clip=False)
        ax.texts[-1].set_x(x)
        ax.texts[-1].set_position((x + 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0]), ty))
    ax.set_ylim(3.28, 3.9)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel("validation loss (nats per token)", color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muon", default=None, help="Muon's log.jsonl (Natural-Muon run), optional")
    ap.add_argument("--out", default="docs/paper/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    series = []
    for label, pattern, colour, dash in SERIES:
        hits = glob.glob(pattern)
        assert len(hits) == 1, (pattern, hits)
        series.append((label, curve(hits[0]), colour, dash, False))
    if args.muon:
        pts = curve(args.muon)
        series.append((MUON[0], pts, MUON[2], MUON[3], pts[-1][0] < 73242))
    plot(series, 0, "optimizer steps (thousands)", out / "val_vs_steps.png", xscale=1000)
    print("wrote", out / "val_vs_steps.png")


if __name__ == "__main__":
    main()
