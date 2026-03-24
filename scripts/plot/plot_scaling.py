#!/usr/bin/env python3
"""
Fig.2-4: Scaling curves

Fig.2: V-Scaling curves (FM/Diff/AR/MLP overlaid)
Fig.3: L-Scaling curves
Fig.4: A-Scaling curves
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


def plot_v_scaling(csv_path: str, system: str = None,
                   output_dir: str = "results/figures"):
    """Fig.2: V-Scaling across action architectures (Group B)."""
    df = pd.read_csv(csv_path)
    df_b = df[df["group"] == "B"]
    if system:
        df_b = df_b[df_b["hardware"] == system]

    if df_b.empty:
        print("No Group B data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    v_order = {"V-S": 0, "V-M": 1, "V-L": 2}
    v_labels = ["SigLIP2-B\n(86M)", "SigLIP2-L\n(307M)", "SigLIP2-So\n(400M)"]

    for ax, metric, ylabel in zip(axes,
                                   ["e2e_time_ms", "e2e_hz"],
                                   ["E2E Latency (ms)", "Throughput (Hz)"]):
        for atype in df_b["action_type"].unique():
            sub = df_b[df_b["action_type"] == atype].copy()
            sub["_vx"] = sub["vision_key"].map(v_order)
            sub = sub.sort_values("_vx")

            ax.plot(sub["_vx"], sub[metric],
                    marker="o", color=ACTION_COLORS.get(atype, "gray"),
                    label=ACTION_LABELS.get(atype, atype), linewidth=2)

        ax.set_xticks(range(3))
        ax.set_xticklabels(v_labels)
        ax.set_xlabel("Vision Encoder")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(f"V-Scaling (fix L=1.5B) — {system or 'All'}", y=1.02)
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"v_scaling_{system or 'all'}.png"
    plt.savefig(out_dir / fname, bbox_inches="tight")
    print(f"Saved {out_dir / fname}")
    plt.close()


def plot_l_scaling(csv_path: str, system: str = None,
                   output_dir: str = "results/figures"):
    """Fig.3: L-Scaling across action architectures (Group C + A middle)."""
    df = pd.read_csv(csv_path)
    # Group C has V=L (SigLIP2-L) with L-S and L-L
    # Group A configs 4,5,6 have V-M with L-S/L-M/L-L and FM-M
    df_c = df[df["group"].isin(["C"])]
    if system:
        df_c = df_c[df_c["hardware"] == system]

    if df_c.empty:
        print("No Group C data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    l_order = {"L-S": 0, "L-M": 1, "L-L": 2}
    l_labels = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B"]

    for ax, metric, ylabel in zip(axes,
                                   ["e2e_time_ms", "e2e_hz"],
                                   ["E2E Latency (ms)", "Throughput (Hz)"]):
        for atype in df_c["action_type"].unique():
            sub = df_c[df_c["action_type"] == atype].copy()
            sub["_lx"] = sub["language_key"].map(l_order)
            sub = sub.sort_values("_lx")

            ax.plot(sub["_lx"], sub[metric],
                    marker="s", color=ACTION_COLORS.get(atype, "gray"),
                    label=ACTION_LABELS.get(atype, atype), linewidth=2)

        ax.set_xticks(range(len(l_labels)))
        ax.set_xticklabels(l_labels)
        ax.set_xlabel("Language Backbone")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle(f"L-Scaling (fix V=SigLIP2-L) — {system or 'All'}", y=1.02)
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"l_scaling_{system or 'all'}.png"
    plt.savefig(out_dir / fname, bbox_inches="tight")
    print(f"Saved {out_dir / fname}")
    plt.close()


def plot_a_scaling(csv_path: str, system: str = None,
                   output_dir: str = "results/figures"):
    """Fig.4: A-Scaling on VLM-5 (Group D)."""
    df = pd.read_csv(csv_path)
    df_d = df[df["group"] == "D"]
    if system:
        df_d = df_d[df_d["hardware"] == system]

    if df_d.empty:
        print("No Group D data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Map action keys to param sizes for x-axis
    action_params = {
        "FM-S": 50, "FM-M": 200, "FM-L": 450,
        "Diff-S": 50, "Diff-M": 200, "Diff-L": 450,
        "MLP-S": 10, "MLP-M": 30, "MLP-L": 80,
    }

    for atype_prefix, color, label in [
        ("FM", "#2196F3", "Flow Matching"),
        ("Diff", "#FF9800", "Diffusion"),
        ("MLP", "#F44336", "MLP"),
    ]:
        sub = df_d[df_d["action_key"].str.startswith(atype_prefix)].copy()
        if sub.empty:
            continue
        sub["_params"] = sub["action_key"].map(action_params)
        sub = sub.sort_values("_params")

        ax.plot(sub["_params"], sub["e2e_time_ms"],
                marker="D", color=color, label=label, linewidth=2)

    ax.set_xlabel("Action Head Parameters (M)")
    ax.set_ylabel("E2E Latency (ms)")
    ax.set_title(f"A-Scaling on VLM-5 (SigLIP2-L + Qwen2.5-1.5B) — {system or 'All'}")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"a_scaling_{system or 'all'}.png"
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

    plot_v_scaling(args.csv, args.system, args.output_dir)
    plot_l_scaling(args.csv, args.system, args.output_dir)
    plot_a_scaling(args.csv, args.system, args.output_dir)
