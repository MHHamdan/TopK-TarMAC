"""Plot learning curves from results/<run>/seed_*/metrics.csv files.

Usage:
    python scripts/plot_curves.py --runs run_a run_b --labels "Method A" "Method B"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_run(run: str) -> pd.DataFrame:
    seed_dirs = sorted((RESULTS / run).glob("seed_*"))
    frames = []
    for d in seed_dirs:
        if not (d / "metrics.csv").exists():
            continue
        df = pd.read_csv(d / "metrics.csv")
        df["seed"] = int(d.name.removeprefix("seed_"))
        df["run"] = run
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No metrics.csv under {RESULTS / run}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--metric", default="ep_return_mean")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()
    labels = args.labels or args.runs

    fig, ax = plt.subplots(figsize=(6, 4))
    for run, lbl in zip(args.runs, labels):
        df = load_run(run)
        # Bucket by global_step. Use 50-bucket evenly spaced average + std.
        max_step = df["global_step"].max()
        bins = np.linspace(0, max_step, 51)
        df["bucket"] = pd.cut(df["global_step"], bins, labels=False, include_lowest=True)
        agg = (df.groupby(["bucket", "seed"])[args.metric].mean().reset_index()
                 .groupby("bucket")[args.metric].agg(["mean", "std", "count"]))
        x = bins[:-1] + np.diff(bins) / 2
        x = x[: len(agg)]
        m, s = agg["mean"].values, agg["std"].fillna(0).values
        ax.plot(x, m, label=f"{lbl} (n={int(agg['count'].max())})")
        ax.fill_between(x, m - s, m + s, alpha=0.2)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(args.metric.replace("_", " "))
    if args.title:
        ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = args.out or "paper/figures/curve_default.pdf"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
