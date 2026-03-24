#!/usr/bin/env python3
"""
Fig.6: Component-level latency breakdown (stacked bar chart)

For all 64 configurations, shows Vision / VLM / Action time breakdown.
Grouped by action architecture (FM/Diff/AR/MLP).
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.rcParams.update({"font.size": 10, "figure.dpi": 150})

COLORS = {
    "vision": "#4ECDC4",
    "vlm": "#FF6B6B",
    "action": "#45B7D1",
}


def plot_breakdown(csv_path: str, system: str = None, output_dir: str = "results/figures"):
    """Plot component-level latency breakdown."""
    df = pd.read_csv(csv_path)

    if system:
        df = df[df["hardware"] == system]

    if df.empty:
        print(f"No data for system={system}")
        return

    # Sort by action type then config_id
    action_order = {"flow_matching": 0, "diffusion": 1, "autoregressive": 2, "mlp": 3}
    df["_sort"] = df["action_type"].map(action_order)
    df = df.sort_values(["_sort", "config_id"])

    fig, ax = plt.subplots(figsize=(20, 6))

    x = np.arange(len(df))
    width = 0.7

    ax.bar(x, df["vision_time_ms"], width, label="Vision Encoder",
           color=COLORS["vision"])
    ax.bar(x, df["vlm_time_ms"], width, bottom=df["vision_time_ms"],
           label="Language Backbone", color=COLORS["vlm"])
    ax.bar(x, df["action_time_ms"], width,
           bottom=df["vision_time_ms"] + df["vlm_time_ms"],
           label="Action Head", color=COLORS["action"])

    # Add action type separators
    prev_type = None
    for i, (_, row) in enumerate(df.iterrows()):
        if prev_type is not None and row["action_type"] != prev_type:
            ax.axvline(x=i - 0.5, color="gray", linestyle="--", alpha=0.5)
        prev_type = row["action_type"]

    ax.set_xlabel("VLA Configuration")
    ax.set_ylabel("Latency (ms)")
    system_name = system or "All Systems"
    ax.set_title(f"Component-Level Latency Breakdown — {system_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{int(r)}" for r in df["config_id"]], rotation=90, fontsize=6)
    ax.legend()

    # Add Hz thresholds
    for hz, color in [(10, "red"), (50, "orange"), (100, "green")]:
        ax.axhline(y=1000/hz, color=color, linestyle=":", alpha=0.5,
                   label=f"{hz} Hz")

    ax.legend(loc="upper left")
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"breakdown_{system or 'all'}.png"
    plt.savefig(out_dir / fname, bbox_inches="tight")
    plt.savefig(out_dir / fname.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {out_dir / fname}")
    plt.close()


def plot_breakdown_by_action_type(csv_path: str, system: str = None,
                                   output_dir: str = "results/figures"):
    """Plot breakdown grouped by action type with component ratios."""
    df = pd.read_csv(csv_path)
    if system:
        df = df[df["hardware"] == system]
    if df.empty:
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    action_types = ["flow_matching", "diffusion", "autoregressive", "mlp"]
    action_labels = ["Flow Matching", "Diffusion", "Autoregressive", "MLP"]

    for ax, atype, alabel in zip(axes, action_types, action_labels):
        sub = df[df["action_type"] == atype].sort_values("config_id")
        if sub.empty:
            ax.set_title(f"{alabel}\n(no data)")
            continue

        x = np.arange(len(sub))
        total = sub["e2e_time_ms"]

        # Stacked bars showing percentage
        v_pct = sub["vision_time_ms"] / total * 100
        l_pct = sub["vlm_time_ms"] / total * 100
        a_pct = sub["action_time_ms"] / total * 100

        ax.bar(x, v_pct, color=COLORS["vision"], label="Vision")
        ax.bar(x, l_pct, bottom=v_pct, color=COLORS["vlm"], label="LLM")
        ax.bar(x, a_pct, bottom=v_pct + l_pct, color=COLORS["action"], label="Action")

        ax.set_title(f"{alabel} (n={len(sub)})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"#{int(r)}" for r in sub["config_id"]], rotation=90, fontsize=7)
        if ax == axes[0]:
            ax.set_ylabel("Latency Share (%)")

    axes[0].legend(loc="lower left")
    plt.suptitle(f"Latency Breakdown by Action Type — {system or 'All'}", y=1.02)
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"breakdown_by_action_{system or 'all'}.png"
    plt.savefig(out_dir / fname, bbox_inches="tight")
    print(f"Saved {out_dir / fname}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/vla_benchmark_results.csv")
    parser.add_argument("--system", default=None)
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    plot_breakdown(args.csv, args.system, args.output_dir)
    plot_breakdown_by_action_type(args.csv, args.system, args.output_dir)
