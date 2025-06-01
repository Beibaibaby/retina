#!/usr/bin/env python3
"""
Monte-Carlo batch driver – mean ± SD shading.
Usage:
    python scripts/run_batch.py -N 50 --seed 42 --out figs/batch_50.png
"""
from pathlib import Path
import argparse
from sac_analysis.batch import run_batch
from sac_analysis.viz  import plot_traces_grid

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", "--num-runs", type=int, default=20, help="number of repeats")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--out", type=str, default="figs/batch.png", help="output figure")
    args = ap.parse_args()

    t, summary = run_batch(args.num_runs, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_traces_grid(t, summary,
                     title="Two-SAC model",
                     out_path=out_path,
                     N_runs=args.num_runs)
    print(f"✓ saved → {out_path}")
