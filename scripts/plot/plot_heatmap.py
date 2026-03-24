#!/usr/bin/env python3
"""
Fig.1: 3×3 V×L Heatmap (latency or Hz)

Shows how latency varies across the VLM matrix for a fixed action head.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.rcParams.update({"font.size": 11, "figure.dpi": 150})


def plot_vl_heatmap(csv_path: str, system: str = None,
                    metric: str = "e2e_hz",
                    output_dir: str = "results/figures"):
    """Plot 3×3 V×L heatmap for Group A configs (fixed FM-M action)."""
    df = pd.read_csv(csv_path)

    # Filter to Group A (3×3 grid, FM-M fixed)
    df_a = df[df["group"] == "A"]
    if system:
        df_a = df_a[df_a["hardware"] == system]

    if df_a.empty:
        print("No Group A data found")
        return

    # Build heatmap matrix
    v_labels = ["SigLIP2-B\n(86M)", "SigLIP2-L\n(307M)", "SigLIP2-So\n(400M)"]
    l_labels = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B"]
    v_keys = ["V-S", "V-M", "V-L"]
    l_keys = ["L-S", "L-M", "L-L"]

    matrix = np.zeros((3, 3))
    for i, vk in enumerate(v_keys):
        for j, lk in enumerate(l_keys):
            row = df_a[(df_a["vision_key"] == vk) & (df_a["language_key"] == lk)]
            if not row.empty:
                matrix[i, j] = row[metric].values[0]

    fig, ax = plt.subplots(figsize=(8, 6))

    metric_label = {
        "e2e_hz": "Inference Throughput (Hz)",
        "e2e_time_ms": "E2E Latency (ms)",
        "vlm_time_ms": "VLM Latency (ms)",
    }.get(metric, metric)

    cmap = "YlOrRd_r" if "hz" in metric.lower() else "YlOrRd"
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap=cmap,
                xticklabels=l_labels, yticklabels=v_labels,
                ax=ax, cbar_kws={"label": metric_label})

    ax.set_xlabel("Language Backbone")
    ax.set_ylabel("Vision Encoder")
    system_name = system or "All"
    ax.set_title(f"V×L Grid — {metric_label} ({system_name})")

    plt.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"heatmap_vl_{metric}_{system or 'all'}.png"
    plt.savefig(out_dir / fname, bbox_inches="tight")
    plt.savefig(out_dir / fname.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {out_dir / fname}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/vla_benchmark_results.csv")
    parser.add_argument("--system", default=None)
    parser.add_argument("--metric", default="e2e_hz",
                        choices=["e2e_hz", "e2e_time_ms", "vlm_time_ms"])
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    plot_vl_heatmap(args.csv, args.system, args.metric, args.output_dir)
