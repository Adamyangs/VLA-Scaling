#!/usr/bin/env python3
"""
Controlled-variable comparison figures for VLA architecture analysis.

Fig 1: Fix VLM-5, compare 4 action types (FM-M vs Diff-M vs AR vs MLP-M)
       → per-component stacked bar, both platforms side by side
Fig 2: Fix VLM-5, action head size scaling (S/M/L) per architecture
       → grouped bar chart showing how action head size affects E2E
Fig 3: Fix action head (FM-M), vary VLM backbone (9 VLMs)
       → stacked bar sorted by total latency, showing V/L/A breakdown
Fig 4: Fix action type, V-Scaling (3 vision encoders, L fixed)
       → line plot, one line per action type, both platforms
Fig 5: Fix action type, L-Scaling (3 language backbones, V fixed)
       → line plot, one line per action type, both platforms
Fig 6: Chunk size & denoising steps trade-off (FM vs AR)
       → dual-axis or multi-panel showing FM chunk/steps and AR chunk scaling
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.rcParams.update({
    "font.size": 11,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Consistent colors
C_VISION = "#2EC4B6"
C_VLM = "#E71D36"
C_ACTION = "#3772FF"

# High-contrast, colorblind-friendly palette
ACTION_COLORS = {
    "Cascade": "#E63946",     # red
    "SharedAttn": "#457B9D",  # steel blue
    "CrossAttn": "#F4A261",   # orange
    "AR-Naive": "#2A9D8F",    # teal
    "AR-FAST": "#264653",     # dark teal
    "Regress": "#A8DADC",     # light blue
}

ACTION_MARKERS = {
    "Cascade": "o",
    "SharedAttn": "s",
    "CrossAttn": "D",
    "AR-Naive": "^",
    "AR-FAST": "v",
    "Regress": "P",
}

ACTION_DISPLAY = {
    "FM-S": "FM-S\n(50M)", "FM-M": "FM-M\n(200M)", "FM-L": "FM-L\n(450M)",
    "Diff-S": "Diff-S\n(50M)", "Diff-M": "Diff-M\n(200M)", "Diff-L": "Diff-L\n(450M)",
    "AR": "AR\n(shared)", "MLP-S": "MLP-S\n(10M)", "MLP-M": "MLP-M\n(30M)", "MLP-L": "MLP-L\n(80M)",
}

HW_SHORT = {
    "Jetson_AGX_Orin_64GB": "Orin 64GB",
    "A100_80GB": "A100 80GB",
    "A800_80GB": "A800 80GB",
    "Jetson_AGX_Thor": "Thor",
    "H100": "H100",
    "RTX_4090": "RTX 4090",
}


def load(csv_path):
    return pd.read_csv(csv_path)


def filter_baseline(df):
    """Keep only baseline configs: default chunk_size=10, default denoising_steps.

    Removes Group G/H/I/K/L sweep configs that override chunk or steps.
    Only keeps groups A-E, M (baseline defaults) to avoid duplicates.
    """
    baseline_groups = ["A", "B", "C", "D", "E", "M"]  # exclude sweep groups G/H/I/K/L
    return df[df["group"].isin(baseline_groups)]


# ================================================================
# Fig 1: Fix VLM-5, compare 4 action architectures (M size)
# ================================================================
def fig1_action_type_comparison(df, output_dir):
    """Side-by-side bar: 4 action types × 2 platforms, per-component breakdown."""
    action_keys = ["Cascade-M", "SharedAttn-M", "CrossAttn-M", "AR-Naive", "AR-FAST", "Regress-M"]
    action_labels = [
        "Cascade\nDenoise\n(200M DiT)",
        "SharedAttn\nDenoise\n(VLM shared)",
        "CrossAttn\nDenoise\n(200M DiT)",
        "AR-Naive\n(per-dim\ntokens)",
        "AR-FAST\n(DCT+BPE\ncompressed)",
        "Direct\nRegression\n(30M MLP)",
    ]
    systems = sorted(df["hardware"].unique())

    base = filter_baseline(df)
    sub = base[(base["vision_key"] == "V-M") & (base["language_key"] == "L-M") &
               (base["action_key"].isin(action_keys))]

    fig, axes = plt.subplots(1, len(systems), figsize=(6 * len(systems), 5.5), sharey=False)
    if len(systems) == 1:
        axes = [axes]

    for ax, hw in zip(axes, systems):
        hw_data = sub[sub["hardware"] == hw]
        x = np.arange(len(action_keys))
        width = 0.55

        v_vals, l_vals, a_vals = [], [], []
        for ak in action_keys:
            row = hw_data[hw_data["action_key"] == ak]
            if row.empty:
                v_vals.append(0); l_vals.append(0); a_vals.append(0)
            else:
                r = row.iloc[0]
                v_vals.append(r["vision_time_ms"])
                l_vals.append(r["vlm_time_ms"])
                a_vals.append(r["action_time_ms"])

        v_arr = np.array(v_vals)
        l_arr = np.array(l_vals)
        a_arr = np.array(a_vals)

        bars_v = ax.bar(x, v_arr, width, label="Vision Encoder", color=C_VISION)
        bars_l = ax.bar(x, l_arr, width, bottom=v_arr, label="Language Backbone", color=C_VLM)
        bars_a = ax.bar(x, a_arr, width, bottom=v_arr + l_arr, label="Action Head", color=C_ACTION)

        # Annotate E2E on top
        for i, (v, l, a) in enumerate(zip(v_arr, l_arr, a_arr)):
            total = v + l + a
            ax.text(i, total + total * 0.02, f"{total:.1f}ms\n({1000/total:.0f}Hz)",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(action_labels, fontsize=9)
        ax.set_ylabel("Latency (ms)")
        ax.set_title(HW_SHORT[hw], fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Action Head Architecture Comparison (VLM-5: SigLIP2-L + Qwen2.5-1.5B, chunk=10)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, output_dir, "fig1_action_type_comparison")


# ================================================================
# Fig 2: Fix VLM-5, action head size scaling (S/M/L)
# ================================================================
def fig2_action_size_scaling(df, output_dir):
    """Grouped bar: S/M/L for FM, Diff, MLP on VLM-5."""
    systems = sorted(df["hardware"].unique())
    arch_groups = [
        ("Cascade Denoise", ["Cascade-S", "Cascade-M", "Cascade-L"], ["S\n(50M)", "M\n(200M)", "L\n(450M)"]),
        ("SharedAttn Denoise", ["SharedAttn-S", "SharedAttn-M", "SharedAttn-L"], ["S\n(50M)", "M\n(200M)", "L\n(450M)"]),
        ("Direct Regression", ["Regress-S", "Regress-M", "Regress-L"], ["S\n(10M)", "M\n(30M)", "L\n(80M)"]),
    ]

    systems = sorted(df["hardware"].unique())
    base = filter_baseline(df)
    sub = base[(base["vision_key"] == "V-M") & (base["language_key"] == "L-M")]

    fig, axes = plt.subplots(len(systems), 3, figsize=(16, 4.5 * len(systems)), sharey="row")
    if len(systems) == 1:
        axes = [axes]

    for col, (arch_name, keys, labels) in enumerate(arch_groups):
        for row, hw in enumerate(systems):
            ax = axes[row][col]
            hw_data = sub[sub["hardware"] == hw]

            x = np.arange(len(keys))
            width = 0.5

            v_vals, l_vals, a_vals = [], [], []
            for ak in keys:
                r = hw_data[hw_data["action_key"] == ak]
                if r.empty:
                    v_vals.append(0); l_vals.append(0); a_vals.append(0)
                else:
                    r = r.iloc[0]
                    v_vals.append(r["vision_time_ms"])
                    l_vals.append(r["vlm_time_ms"])
                    a_vals.append(r["action_time_ms"])

            v_arr, l_arr, a_arr = np.array(v_vals), np.array(l_vals), np.array(a_vals)
            ax.bar(x, v_arr, width, color=C_VISION, label="Vision")
            ax.bar(x, l_arr, width, bottom=v_arr, color=C_VLM, label="LLM")
            ax.bar(x, a_arr, width, bottom=v_arr + l_arr, color=C_ACTION, label="Action")

            for i, (v, l, a) in enumerate(zip(v_arr, l_arr, a_arr)):
                total = v + l + a
                if total > 0:
                    ax.text(i, total + total * 0.02, f"{total:.1f}ms",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            if row == 0:
                ax.set_title(arch_name, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{HW_SHORT[hw]}\nLatency (ms)", fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle("Action Head Size Scaling (VLM-5: SigLIP2-L + Qwen2.5-1.5B)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, output_dir, "fig2_action_size_scaling")


# ================================================================
# Fig 3: Fix FM-M action, compare all 9 VLM backbones
# ================================================================
def fig3_vlm_backbone_comparison(df, output_dir):
    """Stacked bar for 9 VLMs with FM-M, showing V/L/A component ratio."""
    systems = sorted(df["hardware"].unique())
    sub = df[(df["action_key"] == "Cascade-M") & (df["group"] == "A")]

    vlm_labels = [
        "B+0.5B", "B+1.5B", "B+3B",
        "L+0.5B", "L+1.5B", "L+3B",
        "So+0.5B", "So+1.5B", "So+3B",
    ]

    fig, axes = plt.subplots(1, len(systems), figsize=(5.5 * len(systems), 5.5), sharey=False)
    if len(systems) == 1:
        axes = [axes]

    for ax, hw in zip(axes, systems):
        hw_data = sub[sub["hardware"] == hw].sort_values("config_id")
        if hw_data.empty:
            continue

        x = np.arange(len(hw_data))
        width = 0.6

        v_arr = hw_data["vision_time_ms"].values
        l_arr = hw_data["vlm_time_ms"].values
        a_arr = hw_data["action_time_ms"].values

        ax.bar(x, v_arr, width, color=C_VISION, label="Vision Encoder")
        ax.bar(x, l_arr, width, bottom=v_arr, color=C_VLM, label="Language Backbone")
        ax.bar(x, a_arr, width, bottom=v_arr + l_arr, color=C_ACTION, label="Action Head (Cascade-M)")

        for i, (v, l, a) in enumerate(zip(v_arr, l_arr, a_arr)):
            total = v + l + a
            ax.text(i, total + total * 0.01, f"{1000/total:.0f}Hz",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(vlm_labels, fontsize=8.5, rotation=0)
        ax.set_xlabel("Vision + Language Backbone")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(HW_SHORT[hw], fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)

        # Add vision encoder group separators
        for sep in [2.5, 5.5]:
            ax.axvline(x=sep, color="gray", linestyle="--", alpha=0.3)
        # Group labels at bottom
        for gx, glabel in [(1, "SigLIP2-B"), (4, "SigLIP2-L"), (7, "SigLIP2-So")]:
            ax.text(gx, -ax.get_ylim()[1] * 0.08, glabel,
                    ha="center", fontsize=8, color="gray", style="italic")

    fig.suptitle("VLM Backbone Comparison (Fixed Action: FM-M 200M, chunk=10, steps=10)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, output_dir, "fig3_vlm_backbone_comparison")


# ================================================================
# Fig 4: V-Scaling — fix L=1.5B, sweep V across 4 action types
# ================================================================
def fig4_v_scaling(df, output_dir):
    """Line plot: V-Scaling per action type, both platforms."""
    systems = sorted(df["hardware"].unique())
    # Group B (V-Scaling x Diff/AR/MLP) + Group A row 2 (V-M x FM-M with L=1.5B)
    # We need FM-M with L=1.5B for all 3 V sizes → configs 2,5,8 from Group A
    action_items = [
        ("Cascade-M", "Cascade"), ("SharedAttn-M", "SharedAttn"),
        ("CrossAttn-M", "CrossAttn"), ("AR-Naive", "AR-Naive"),
        ("AR-FAST", "AR-FAST"), ("Regress-M", "Regress"),
    ]

    base = filter_baseline(df)
    sub = base[(base["language_key"] == "L-M") &
               (base["action_key"].isin([a[0] for a in action_items]))]

    v_order = {"V-S": 0, "V-M": 1, "V-L": 2}
    v_labels = ["SigLIP2-B\n(86M)", "SigLIP2-L\n(307M)", "SigLIP2-So\n(400M)"]

    fig, axes = plt.subplots(1, len(systems), figsize=(5.5 * len(systems), 5.5))
    if len(systems) == 1:
        axes = [axes]

    for ax, hw in zip(axes, systems):
        hw_data = sub[sub["hardware"] == hw]

        for ak, color_key in action_items:
            a_data = hw_data[hw_data["action_key"] == ak].copy()
            a_data["_vx"] = a_data["vision_key"].map(v_order)
            a_data = a_data.sort_values("_vx")
            if a_data.empty:
                continue

            marker = ACTION_MARKERS.get(color_key, "o")
            # Offset CrossAttn slightly so it doesn't hide behind Cascade
            offset = 0.05 if color_key == "CrossAttn" else 0
            ax.plot(a_data["_vx"] + offset, a_data["e2e_time_ms"],
                    marker=marker, color=ACTION_COLORS[color_key],
                    label=ak, linewidth=2.5, markersize=8)

            for _, row in a_data.iterrows():
                ax.annotate(f"{row['e2e_time_ms']:.1f}",
                            (row["_vx"], row["e2e_time_ms"]),
                            textcoords="offset points", xytext=(8, 4),
                            fontsize=7.5, color=ACTION_COLORS[color_key])

        ax.set_xticks(range(3))
        ax.set_xticklabels(v_labels)
        ax.set_xlabel("Vision Encoder (fixed L = Qwen2.5-1.5B)")
        ax.set_ylabel("E2E Latency (ms)")
        ax.set_title(HW_SHORT[hw], fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    fig.suptitle("Vision Encoder Scaling (Fixed L = Qwen2.5-1.5B, chunk=10)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, output_dir, "fig4_v_scaling")


# ================================================================
# Fig 5: L-Scaling — fix V=SigLIP2-L, sweep L across 4 action types
# ================================================================
def fig5_l_scaling(df, output_dir):
    """Line plot: L-Scaling per action type, both platforms."""
    systems = sorted(df["hardware"].unique())
    action_items = [
        ("Cascade-M", "Cascade"), ("SharedAttn-M", "SharedAttn"),
        ("CrossAttn-M", "CrossAttn"), ("AR-Naive", "AR-Naive"),
        ("AR-FAST", "AR-FAST"), ("Regress-M", "Regress"),
    ]

    base = filter_baseline(df)
    sub = base[(base["vision_key"] == "V-M") &
               (base["action_key"].isin([a[0] for a in action_items]))]

    l_order = {"L-S": 0, "L-M": 1, "L-L": 2}
    l_labels = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B"]

    fig, axes = plt.subplots(1, len(systems), figsize=(5.5 * len(systems), 5.5))
    if len(systems) == 1:
        axes = [axes]

    for ax, hw in zip(axes, systems):
        hw_data = sub[sub["hardware"] == hw]

        for ak, color_key in action_items:
            a_data = hw_data[hw_data["action_key"] == ak].copy()
            a_data["_lx"] = a_data["language_key"].map(l_order)
            a_data = a_data.sort_values("_lx")
            if a_data.empty:
                continue

            n_points = len(a_data)
            marker = ACTION_MARKERS.get(color_key, "o")
            offset = 0.03 if color_key == "CrossAttn" else 0
            ax.plot(a_data["_lx"] + offset, a_data["e2e_time_ms"],
                    marker=marker, color=ACTION_COLORS[color_key],
                    label=ak, linewidth=2.5, markersize=8)

            for _, row in a_data.iterrows():
                ax.annotate(f"{row['e2e_time_ms']:.1f}",
                            (row["_lx"], row["e2e_time_ms"]),
                            textcoords="offset points", xytext=(8, 4),
                            fontsize=7.5, color=ACTION_COLORS[color_key])

        ax.set_xticks(range(3))
        ax.set_xticklabels(l_labels)
        ax.set_xlabel("Language Backbone (fixed V = SigLIP2-L)")
        ax.set_ylabel("E2E Latency (ms)")
        ax.set_title(HW_SHORT[hw], fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    fig.suptitle("Language Backbone Scaling (Fixed V = SigLIP2-L, chunk=10)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, output_dir, "fig5_l_scaling")


# ================================================================
# Fig 6: Chunk size & denoising steps trade-off
# ================================================================
def fig6_chunk_steps_tradeoff(df, output_dir):
    """Multi-panel: Denoise chunk, Denoise steps, AR-Naive vs AR-FAST chunk."""
    systems = sorted(df["hardware"].unique())

    fig, axes = plt.subplots(len(systems), 3, figsize=(16, 4.5 * len(systems)))

    for row, hw in enumerate(systems):
        hw_data = df[df["hardware"] == hw]

        # Panel 1: Cascade-M chunk size sweep (Group G) — line plot
        ax = axes[row][0] if len(systems) > 1 else axes[0]
        g_data = hw_data[hw_data["group"] == "G"].sort_values("chunk_size")
        if not g_data.empty:
            ax.plot(g_data["chunk_size"], g_data["action_time_ms"],
                    marker="o", color=ACTION_COLORS["Cascade"], linewidth=2, label="Action (Cascade-M)")
            vlm_base = g_data["vlm_time_ms"].iloc[0]
            ax.axhline(y=vlm_base, color=C_VLM, linestyle="--", alpha=0.7,
                       label=f"VLM ({vlm_base:.1f}ms)")
            for _, r in g_data.iterrows():
                ax.annotate(f"{r['action_time_ms']:.1f}", (r["chunk_size"], r["action_time_ms"]),
                            textcoords="offset points", xytext=(5, 5), fontsize=7)
            ax.set_xscale("log")
            ax.set_xlabel("Chunk Size")
        if row == 0:
            ax.set_title("Denoise: Chunk Sweep", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{HW_SHORT.get(hw, hw)}\nAction Latency (ms)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

        # Panel 2: Cascade-M steps sweep (Group H) — line plot
        ax = axes[row][1] if len(systems) > 1 else axes[1]
        h_data = hw_data[hw_data["group"] == "H"].sort_values("denoising_steps")
        # Add default config (steps=10)
        default_cascade = hw_data[(hw_data["action_key"] == "Cascade-M") &
                                   (hw_data["group"] == "A") &
                                   (hw_data["vision_key"] == "V-M") &
                                   (hw_data["language_key"] == "L-M")]
        h_all = pd.concat([default_cascade, h_data]).drop_duplicates("denoising_steps").sort_values("denoising_steps")
        if not h_all.empty:
            ax.plot(h_all["denoising_steps"], h_all["action_time_ms"],
                    marker="s", color=ACTION_COLORS["Cascade"], linewidth=2, label="Action (Cascade-M)")
            vlm_base = h_all["vlm_time_ms"].iloc[0]
            ax.axhline(y=vlm_base, color=C_VLM, linestyle="--", alpha=0.7,
                       label=f"VLM ({vlm_base:.1f}ms)")
            for _, r in h_all.iterrows():
                ax.annotate(f"{r['action_time_ms']:.1f}", (r["denoising_steps"], r["action_time_ms"]),
                            textcoords="offset points", xytext=(5, 5), fontsize=7)
            ax.set_xscale("log")
            ax.set_xlabel("Denoising Steps")
        if row == 0:
            ax.set_title("Denoise: Steps Sweep", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

        # Panel 3: AR-Naive vs AR-FAST chunk sweep — line plot, log-log
        ax = axes[row][2] if len(systems) > 1 else axes[2]
        ar_i = hw_data[hw_data["group"] == "I"]
        ar_naive = ar_i[ar_i["action_type"] == "ar_naive"].sort_values("chunk_size")
        ar_fast = ar_i[ar_i["action_type"] == "ar_fast"].sort_values("chunk_size")

        if not ar_naive.empty:
            ax.plot(ar_naive["chunk_size"], ar_naive["action_time_ms"],
                    marker="^", color=ACTION_COLORS["AR-Naive"], linewidth=2, label="AR-Naive")
            for _, r in ar_naive.iterrows():
                ax.annotate(f"{r['action_time_ms']:.0f}", (r["chunk_size"], r["action_time_ms"]),
                            textcoords="offset points", xytext=(5, 5), fontsize=7)
        if not ar_fast.empty:
            ax.plot(ar_fast["chunk_size"], ar_fast["action_time_ms"],
                    marker="v", color=ACTION_COLORS["AR-FAST"], linewidth=2, label="AR-FAST")
            for _, r in ar_fast.iterrows():
                ax.annotate(f"{r['action_time_ms']:.0f}", (r["chunk_size"], r["action_time_ms"]),
                            textcoords="offset points", xytext=(5, -12), fontsize=7)
        if not ar_naive.empty:
            vlm_base = ar_naive["vlm_time_ms"].iloc[0]
            ax.axhline(y=vlm_base, color=C_VLM, linestyle="--", alpha=0.7,
                       label=f"VLM ({vlm_base:.1f}ms)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Chunk Size")
        if row == 0:
            ax.set_title("AR-Naive vs AR-FAST", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2, which="both")

    fig.suptitle("Chunk Size & Denoising Steps Trade-off (VLM-5: SigLIP2-L + Qwen2.5-1.5B)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, output_dir, "fig6_chunk_steps_tradeoff")


def save(fig, output_dir, name):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    print(f"  Saved {name}.png + .pdf")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/vla_benchmark_results.csv")
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    df = load(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    fig1_action_type_comparison(df, args.output_dir)
    fig2_action_size_scaling(df, args.output_dir)
    fig3_vlm_backbone_comparison(df, args.output_dir)
    fig4_v_scaling(df, args.output_dir)
    fig5_l_scaling(df, args.output_dir)
    fig6_chunk_steps_tradeoff(df, args.output_dir)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
