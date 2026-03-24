# VLA-Perf++: Roofline Benchmark for Vision-Language-Action Models

An analytical performance benchmark for **93 VLA model configurations**, systematically evaluating how different Vision Encoder, Language Backbone, and Action Head topologies affect inference latency on edge and server hardware.

Built on [NVIDIA VLA-Perf](https://arxiv.org/abs/2602.18397) and [GenZ LLM Analyzer](https://github.com/abhibambhaniya/GenZ-LLM-Analyzer).

## Research Questions & Experiments

We address **7 core questions** about VLA architecture design. Each question maps to specific controlled-variable experiments:

### Q1: Component Scaling — V/L/A 增大谁的延迟收益最高？

> **实验:** 固定其中两个组件，逐步增大第三个，对比 E2E 延迟变化。
> **方法:** Group B (V-Scaling × 6 topologies), Group C (L-Scaling × 6 topologies)
> **关键发现:** Language backbone 是延迟主导因素 — L 从 0.5B→3B 增加 100%+ 延迟，而 V 从 86M→400M 仅增加 <15%。

<p align="center">
<img src="docs/figures/fig4_v_scaling.png" width="95%">
</p>
<p align="center">
<img src="docs/figures/fig5_l_scaling.png" width="95%">
</p>

### Q2: Optimal Allocation — 固定总参数预算，V/L/A 最优比例是什么？

> **实验:** 3×3 V×L 网格（固定 Cascade-M action head），9 种 VLM 组合。
> **方法:** Group A (9 configs)
> **关键发现:** 预算有限时，优先分配给 L（Language backbone），V 选型对延迟影响极小。

<p align="center">
<img src="docs/figures/fig3_vlm_backbone_comparison.png" width="95%">
</p>

### Q3: Bottleneck — 三组件之间是否存在瓶颈效应？

> **实验:** 分析 Group A 九宫格中各组件延迟占比。
> **关键发现:** Edge 上 action head 是瓶颈（占 E2E 的 47-97%），Server 上 VLM prefill 成为瓶颈。

### Q4: Action Architecture — 6 种拓扑，架构 vs 大小哪个更重要？

> **实验:** 固定 VLM-5，对比 6 种 action 拓扑 (Fig 1)；对 Cascade/SharedAttn/Regression 做 S/M/L 尺寸缩放 (Fig 2)。
> **方法:** Group D (size scaling), Group E (corner VLM generalization)
> **关键发现:** 拓扑选择比模型大小重要得多 — Cascade-M(200M) 比 SharedAttn(共享1.5B VLM) 快 3.3×。

<p align="center">
<img src="docs/figures/fig1_action_type_comparison.png" width="95%">
</p>
<p align="center">
<img src="docs/figures/fig2_action_size_scaling.png" width="95%">
</p>

### Q5: Chunk & Steps — chunk size 和 denoising steps 的延迟权衡？

> **实验:** Cascade-M chunk sweep (1→100), steps sweep (1→100), AR-Naive vs AR-FAST chunk sweep (1→50)。
> **方法:** Group G (chunk), Group H (steps), Group I (AR comparison)
> **关键发现:** Denoise 类方法的 action 延迟对 chunk size 近乎常数（并行解码），但对 steps 线性增长。AR-Naive 对 chunk 线性增长，AR-FAST 通过 DCT+BPE 压缩实现 ~5× 加速。

<p align="center">
<img src="docs/figures/fig6_chunk_steps_tradeoff.png" width="95%">
</p>

### Q6: Cross-Platform — 同一架构在 Edge vs Server 上延迟特性有何差异？

> **实验:** 全部 93 configs 在 3 个平台上运行（Orin 64GB / A100 80GB / Thor）。
> **关键发现:** A100 平均比 Orin 快 ~9×；Thor 介于两者之间。Edge 上全部 memory-bound，Server 上 vision encoder 可能 compute-bound。

### Q7: Pareto Frontier — 结合精度数据，什么架构在 accuracy-latency 上最优？

> **实验:** 将 accuracy 数据（如 LIBERO success rate）与延迟数据合并，构建 Pareto 前沿。
> **方法:** `python scripts/plot/plot_pareto.py --accuracy-csv <path>`
> **状态:** 需要实际训练精度数据，benchmark 框架已就绪。

---

Full controlled-variable analysis with 17 insights: **[docs/ANALYSIS.md](docs/ANALYSIS.md)**

## Supported Architectures

### 6 Action Head Topologies

| Type | Connection | Inference | Representative Models |
|------|-----------|-----------|----------------------|
| **Cascade Denoise** | VLM → separate DiT (cross-attn) | DiT × N denoising steps | GR00T N1, CogACT, DexVLA |
| **SharedAttn Denoise** | Action tokens enter VLM self-attn | VLM × N steps (KV cache reused) | pi0, ForceVLA, OneTwoVLA |
| **CrossAttn Denoise** | Separate DiT with cross-attn to VLM | DiT × N steps | SmolVLA, GR-3 |
| **AR-Naive** | VLM outputs per-dim discrete tokens | 1 token × (dof × chunk) | OpenVLA, RT-2 |
| **AR-FAST** | VLM outputs DCT+BPE compressed tokens | 1 token × (dof × chunk / 5) | pi0-FAST |
| **Direct Regression** | VLM hidden → MLP → continuous actions | Single forward pass | OpenVLA-OFT, BridgeVLA |

### Model Components

| Component | Variants |
|-----------|----------|
| **Vision Encoder** | SigLIP2-B (86M), SigLIP2-L (307M), SigLIP2-So (400M) |
| **Language Backbone** | Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B |
| **Denoise Expert** | S (50M), M (200M), L (450M) |
| **MLP Head** | S (10M), M (30M), L (80M) |

### Hardware Platforms

| Platform | Type | BF16 TFLOPS | Memory BW |
|----------|------|-------------|-----------|
| Jetson AGX Orin 64GB | Edge | ~138 (FP16) | 204 GB/s |
| Jetson AGX Thor | Edge | 800 (FP8) | 270 GB/s |
| A100 80GB | Server | 312 | 2039 GB/s |
| H100 SXM | Server | 989 | 3350 GB/s |
| RTX 4090 / 5090 | Desktop | 165 / 209 | 1008 / 1792 GB/s |
| B100 / GB200 | Server | 1750 / 2250 | 8000 GB/s |

Full list: 25+ platforms in GenZ `system_configs.py`.

## Quick Start

```bash
# 1. Install GenZ backend
cd ../vla-perf/genz && pip install -e .

# 2. Install dependencies
cd ../../VLA-Scaling && pip install -r requirements.txt

# 3. Run full benchmark (93 configs × 3 platforms, ~40s)
python scripts/run_benchmark.py

# 4. Generate comparison figures
python scripts/plot/plot_comparison.py
```

### More Examples

```bash
# Run specific phase / group / configs
python scripts/run_benchmark.py --phase P0
python scripts/run_benchmark.py --group D
python scripts/run_benchmark.py --configs 1 5 17

# Different hardware / precision
python scripts/run_benchmark.py --systems H100 RTX_4090
python scripts/run_benchmark.py --bits int8

# Run per-question experiments
python scripts/run_scaling.py --experiment q1
python scripts/run_scaling.py --experiment all

# Pareto frontier (with accuracy data)
python scripts/plot/plot_pareto.py --accuracy-csv path/to/accuracy.csv
```

## Experiment Design (93 configs)

| Phase | Group | Configs | Content | Question |
|-------|-------|:-------:|---------|----------|
| P0 | A | 9 | 3×3 V×L grid × Cascade-M | Q2, Q3 |
| P0 | B | 15 | V-Scaling × 5 topologies (fix L=1.5B) | Q1 |
| P0 | C | 10 | L-Scaling × 5 topologies (fix V=SigLIP2-L) | Q1 |
| P0 | D | 9 | A-Scaling S/M/L × Cascade/SharedAttn/Regress | Q4 |
| P0 | E | 12 | Corner VLMs (1,9) × all 6 topologies | Q4 |
| P1 | G | 7 | Cascade chunk sweep {1,2,5,10,20,50,100} | Q5 |
| P1 | H | 7 | Cascade steps sweep {1,2,5,10,20,50,100} | Q5 |
| P1 | I | 12 | AR-Naive vs AR-FAST chunk {1,2,5,10,20,50} | Q5 |
| P2 | K | 4 | Chunk generalization on VLM-1/9 | Q5 |
| P2 | L | 4 | Steps generalization on VLM-1/9 | Q5 |
| P2 | M | 4 | CrossAttn S/L on corner VLMs | Q4 |

All configs run on 3 platforms (Orin 64GB, A100 80GB, Thor) = **279 data points**.

## Project Structure

```
VLA-Scaling/
├── vla_bench/                   # Core benchmark package
│   ├── configs.py               # 93 VLA configs (Q1-Q7 definitions + registry)
│   ├── engine.py                # VLAPerfEngine: per-component roofline
│   └── network.py               # 8 deployment scenarios
├── scripts/
│   ├── run_benchmark.py         # Main benchmark CLI
│   ├── run_scaling.py           # Per-question experiments
│   └── plot/
│       ├── plot_comparison.py   # 6 controlled-variable figures
│       ├── plot_heatmap.py      # V×L throughput heatmap
│       ├── plot_roofline.py     # Hardware roofline
│       └── plot_pareto.py       # Accuracy-latency frontier
├── docs/
│   ├── ANALYSIS.md              # Full analysis report (17 insights)
│   └── DESIGN.md                # System architecture & methodology
└── results/                     # Benchmark outputs (gitignored)
```

## Acknowledgments

- [NVIDIA VLA-Perf](https://arxiv.org/abs/2602.18397) | [GenZ LLM Analyzer](https://github.com/abhibambhaniya/GenZ-LLM-Analyzer)
- Architecture references: [pi0](https://arxiv.org/abs/2410.24164), [GR00T N1](https://arxiv.org/abs/2503.14734), [SmolVLA](https://arxiv.org/abs/2506.01844), [OpenVLA-OFT](https://openvla-oft.github.io/), [pi0-FAST](https://www.pi.website/research/fast)

## License

Apache 2.0
