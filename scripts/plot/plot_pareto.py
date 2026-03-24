#!/usr/bin/env python3
"""
Fig.11-12: Accuracy-Latency Pareto Frontier

Scatter plot of all 64 VLA configs:
  X: e2e latency (ms) or Hz
  Y: accuracy (success rate) — loaded from accuracy CSV
  Color: action architecture type
  Size: total parameter count

Marks Pareto frontier and Hz threshold lines (10/50/100 Hz).
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 150})

ACTION_COLORS = {
    "flow_matching": "#2196F3",
    "diffusion": "#FF9800",
    "autoregressive": "#4CAF50",
    "mlp": "#F44336",
}

ACTION_LABELS = {
    "flow_matching": "Flow Matching",
    "diffusion": "Diffusion",
    "autoregressive": "Autoregressive",
    "mlp": "MLP",
}


def compute_pareto_frontier(x, y):
    """Compute Pareto frontier indices (minimize x, maximize y)."""
    sorted_indices = np.argsort(x)
    pareto_indices = []
    max_y = -np.inf
    for idx in sorted_indices:
        if y[idx] > max_y:
            pareto_indices.append(idx)
            max_y = y[idx]
    return pareto_indices


def plot_pareto(
    latency_csv: str,
    accuracy_csv: str = None,
    system: str = None,
    output_dir: str = "results/figures",
):
    """
    Plot accuracy-latency Pareto frontier.

    If accuracy_csv is not provided, plots latency-only scatter.
    accuracy_csv should have columns: config_id, accuracy (or success_rate)
    """
    df = pd.read_csv(latency_csv)
    if system:
        df = df[df["hardware"] == system]

    if df.empty:
        print(f"No data for {system}")
        return

    has_accuracy = False
    if accuracy_csv and Path(accuracy_csv).exists():
        acc_df = pd.read_csv(accuracy_csv)
        acc_col = "accuracy" if "accuracy" in acc_df.columns else "success_rate"
        df = df.merge(acc_df[["config_id", acc_col]], on="config_id", how="left")
        has_accuracy = not df[acc_col].isna().all()

    fig, ax = plt.subplots(figsize=(12, 8))

    for atype in df["action_type"].unique():
        sub = df[df["action_type"] == atype]
        color = ACTION_COLORS.get(atype, "gray")
        label = ACTION_LABELS.get(atype, atype)

        if has_accuracy:
            ax.scatter(sub["e2e_time_ms"], sub[acc_col],
                       c=color, s=60, alpha=0.7, label=label, edgecolors="white")
            # Annotate config IDs
            for _, row in sub.iterrows():
                ax.annotate(f"#{int(row['config_id'])}",
                            (row["e2e_time_ms"], row[acc_col]),
                            fontsize=6, alpha=0.7)
        else:
            ax.scatter(sub["e2e_time_ms"], sub["e2e_hz"],
                       c=color, s=60, alpha=0.7, label=label, edgecolors="white")
            for _, row in sub.iterrows():
                ax.annotate(f"#{int(row['config_id'])}",
                            (row["e2e_time_ms"], row["e2e_hz"]),
                            fontsize=6, alpha=0.7)

    # Add Hz threshold lines
    for hz, ls in [(10, "--"), (50, "-."), (100, ":")]:
        ax.axvline(x=1000/hz, color="gray", linestyle=ls, alpha=0.4)
        ax.text(1000/hz, ax.get_ylim()[1] * 0.95, f"{hz}Hz",
                fontsize=8, ha="right", color="gray")

    # Pareto frontier
    if has_accuracy:
        x = df["e2e_time_ms"].values
        y = df[acc_col].values
        valid = ~np.isnan(y)
        if valid.any():
            pareto_idx = compute_pareto_frontier(x[valid], y[valid])
            px = x[valid][pareto_idx]
            py = y[valid][pareto_idx]
            sort_order = np.argsort(px)
            ax.plot(px[sort_order], py[sort_order], "k--", alpha=0.5,
                    label="Pareto Frontier", linewidth=1.5)

    ax.set_xlabel("E2E Latency (ms)")
    ylabel = acc_col.replace("_", " ").title() if has_accuracy else "Throughput (Hz)"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{'Accuracy' if has_accuracy else 'Latency'}-Latency Scatter — {system or 'All'}")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "pareto" if has_accuracy else "latency_scatter"
    fname = f"{suffix}_{system or 'all'}.png"
    plt.savefig(out_dir / fname, bbox_inches="tight")
    plt.savefig(out_dir / fname.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {out_dir / fname}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/vla_benchmark_results.csv")
    parser.add_argument("--accuracy-csv", default=None,
                        help="CSV with config_id and accuracy/success_rate columns")
    parser.add_argument("--system", default=None)
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    plot_pareto(args.csv, args.accuracy_csv, args.system, args.output_dir)
