# VLA-Perf++: Roofline Benchmark for Vision-Language-Action Models

An analytical performance benchmark for **64 VLA model configurations**, systematically evaluating how different Vision Encoder, Language Backbone, and Action Head choices affect inference latency across edge and server hardware.

Built on [NVIDIA VLA-Perf](https://arxiv.org/abs/2602.18397) and [GenZ LLM Analyzer](https://github.com/abhibambhaniya/GenZ-LLM-Analyzer).

## Motivation

Current VLA architecture design relies heavily on intuition. Researchers face unanswered questions:

- Given a total parameter budget and target hardware, how should parameters be allocated across V, L, and A?
- Which action head architecture (Flow Matching / Diffusion / Autoregressive / MLP) is most latency-efficient?
- Where is the latency bottleneck — Vision Encoder, Language Backbone, or Action Head?

NVIDIA's VLA-Perf only covers Pi0 and OpenVLA. **VLA-Perf++ extends it to 64 configurations** covering 3 vision encoders x 3 language backbones x 4 action architectures x 3 sizes, providing component-level latency breakdown to answer these questions.

## Configuration Space

### Model Components

| Component | Variants | Parameters |
|-----------|----------|------------|
| **Vision Encoder** | SigLIP2-B/16, SigLIP2-L/16, SigLIP2-So400m | 86M, 307M, 400M |
| **Language Backbone** | Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B | 0.5B, 1.5B, 3B |
| **Action Head (FM)** | Flow Matching DiT S/M/L | 50M, 200M, 450M |
| **Action Head (Diff)** | Diffusion DiT S/M/L | 50M, 200M, 450M |
| **Action Head (AR)** | Autoregressive (reuses LLM) | 0 extra |
| **Action Head (MLP)** | MLP projection S/M/L | 10M, 30M, 80M |

### VLM Matrix (3x3)

9 Vision-Language backbone combinations used across experiments:

|  | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-3B |
|--|:---:|:---:|:---:|
| **SigLIP2-B** (86M) | VLM-1 (0.6B) | VLM-2 (1.6B) | VLM-3 (3.1B) |
| **SigLIP2-L** (307M) | VLM-4 (0.8B) | VLM-5 (1.8B) | VLM-6 (3.3B) |
| **SigLIP2-So** (400M) | VLM-7 (0.9B) | VLM-8 (1.9B) | VLM-9 (3.4B) |

### 64 Configurations in 3 Phases

| Phase | Configs | Groups | Research Questions |
|-------|---------|--------|-------------------|
| **P0** | 40 | A-F | Q1 (component scaling), Q2 (optimal allocation), Q3 (bottleneck), Q4 (action architecture) |
| **P1** | 12 | G-J | Q5 (chunk size & denoising steps trade-off) |
| **P2** | 12 | K-M | Q5-Q7 supplementary (generalization, quantization) |

## Key Results

All figures below use controlled-variable comparisons: one component varies while others are fixed.

### Fig 1. Action Head Architecture Comparison

Fixed VLM-5 (SigLIP2-L + Qwen2.5-1.5B), compare FM vs Diff vs AR vs MLP at medium size:

<p align="center">
<img src="docs/figures/fig1_action_type_comparison.png" width="90%">
</p>

> AR (Autoregressive) is significantly slower because it generates action tokens sequentially (chunk_size x per-token cost). FM and Diff have identical latency (same transformer arch). MLP is near-free.

### Fig 2. Action Head Size Scaling (S/M/L)

Fixed VLM-5, scaling action head from Small to Large for each architecture:

<p align="center">
<img src="docs/figures/fig2_action_size_scaling.png" width="90%">
</p>

> FM/Diff latency grows significantly with size (50M -> 450M). MLP latency is negligible at all sizes. On edge (Orin), the action head dominates; on server (A800), VLM dominates.

### Fig 3. VLM Backbone Comparison

Fixed action (FM-M), compare all 9 Vision+Language combinations:

<p align="center">
<img src="docs/figures/fig3_vlm_backbone_comparison.png" width="90%">
</p>

> Language backbone (red) is the dominant factor. Scaling V from SigLIP2-B to SigLIP2-So has minimal impact on edge, but on server the action head (blue) becomes the bottleneck.

### Fig 4. Vision Encoder Scaling

Fixed L=Qwen2.5-1.5B, sweep Vision encoder across 4 action types:

<p align="center">
<img src="docs/figures/fig4_v_scaling.png" width="90%">
</p>

> Vision encoder choice barely affects E2E latency — the gap between SigLIP2-B (86M) and SigLIP2-So (400M) is <5ms on edge, <1ms on server. Action head type is the dominant factor.

### Fig 5. Language Backbone Scaling

Fixed V=SigLIP2-L, sweep Language backbone across 4 action types:

<p align="center">
<img src="docs/figures/fig5_l_scaling.png" width="90%">
</p>

> Language backbone scaling has the largest latency impact. AR is especially sensitive because it reuses the LLM for action generation — scaling from 0.5B to 3B increases AR action time proportionally.

### Fig 6. Chunk Size & Denoising Steps Trade-off

FM chunk sweep, FM denoising steps sweep, and AR chunk sweep on VLM-5:

<p align="center">
<img src="docs/figures/fig6_chunk_steps_tradeoff.png" width="90%">
</p>

> FM action latency is nearly constant across chunk sizes (parallel decode) but linear in denoising steps. AR latency scales linearly with chunk size (sequential generation). The red dashed line shows VLM latency as reference — FM stays below it while AR quickly exceeds it.

## Quick Start

### 1. Prerequisites

Clone both repos side-by-side:

```
vla-s/
├── vla-perf/        # NVIDIA VLA-Perf (GenZ backend)
└── VLA-Scaling/     # This project
```

### 2. Install GenZ Backend

```bash
cd ../vla-perf/genz
pip install -e .
```

### 3. Install Dependencies

```bash
cd ../../VLA-Scaling
pip install -r requirements.txt
```

### 4. Run Benchmark

```bash
# All 64 configs on Orin + A800 (takes ~30s)
python scripts/run_benchmark.py

# Specific phase / group / configs
python scripts/run_benchmark.py --phase P0
python scripts/run_benchmark.py --group A
python scripts/run_benchmark.py --configs 1 5 9

# Different hardware
python scripts/run_benchmark.py --systems H100 RTX_4090

# Quantized inference
python scripts/run_benchmark.py --bits int8
```

### 5. Run Scaling Experiments

```bash
# Individual research questions
python scripts/run_scaling.py --experiment q1   # Component scaling
python scripts/run_scaling.py --experiment q2   # V/L allocation
python scripts/run_scaling.py --experiment q4   # Action architecture
python scripts/run_scaling.py --experiment q5   # Chunk / steps sweep

# All experiments
python scripts/run_scaling.py --experiment all
```

### 6. Generate Plots

```bash
python scripts/plot/plot_heatmap.py   --system A800_80GB
python scripts/plot/plot_breakdown.py --system Jetson_AGX_Orin_64GB
python scripts/plot/plot_scaling.py   --system A800_80GB
python scripts/plot/plot_roofline.py  --system A800_80GB
python scripts/plot/plot_pareto.py    --system A800_80GB

# With accuracy data (for Pareto frontier)
python scripts/plot/plot_pareto.py --accuracy-csv path/to/accuracy.csv --system A800_80GB
```

## Inference Pipeline

VLA inference is modeled as 3 sequential stages. Each action head type maps to a different GenZ modeling API:

```
Image(s) ──→ [Vision Encoder] ──→ Visual Tokens
                  (prefill)           │
Prompt   ──→ ────────────────────────┤
                                     ▼
              [Language Backbone] ──→ KV Cache
                  (prefill)           │
                                     ▼
              [Action Head]      ──→ Action Chunk (joint positions × T steps)
```

| Action Type | GenZ Modeling | Scaling Behavior |
|-------------|--------------|------------------|
| **Flow Matching** | `parallel_decode` x N steps | Constant in chunk size, linear in denoising steps |
| **Diffusion** | `parallel_decode` x N steps | Same as FM (identical architecture) |
| **Autoregressive** | `decode` x chunk_size | Linear in chunk size (sequential generation) |
| **MLP** | `prefill` with 1 token | Near-zero latency, no iteration |

## Project Structure

```
VLA-Scaling/
├── vla_bench/                   # Core benchmark package
│   ├── configs.py               # 64 VLA configurations (3 phases, 13 groups)
│   ├── engine.py                # VLAPerfEngine: per-component roofline evaluation
│   └── network.py               # 8 deployment scenarios with network latency
├── scripts/
│   ├── run_benchmark.py         # Main benchmark CLI
│   ├── run_scaling.py           # Per-question scaling experiments (Q1-Q7)
│   └── plot/
│       ├── plot_breakdown.py    # Component latency breakdown (stacked bar)
│       ├── plot_heatmap.py      # V x L throughput heatmap
│       ├── plot_scaling.py      # V/L/A scaling curves
│       ├── plot_roofline.py     # Hardware roofline analysis
│       └── plot_pareto.py       # Accuracy-latency Pareto frontier
├── docs/
│   └── DESIGN.md                # Detailed design document
├── results/                     # Benchmark outputs (gitignored)
└── requirements.txt
```

## Output Format

Results CSV with per-component breakdown:

| Column | Description |
|--------|-------------|
| `vision_time_ms` / `vlm_time_ms` / `action_time_ms` | Per-component latency |
| `e2e_time_ms` | End-to-end latency (V + L + A) |
| `e2e_hz` | Inference throughput (1000 / e2e_time_ms) |
| `vision_boundness` / `vlm_boundness` / `action_boundness` | Compute / Memory / Communication bound |
| `vision_op_intensity` / `vlm_op_intensity` / `action_op_intensity` | Operational intensity (FLOP/Byte) |
| `total_memory_mb` | Total memory footprint (weights + KV cache) |
| `ttfa_ms` | Time to first action (Vision + VLM, before action head) |

## Supported Hardware

| Platform | Type | BF16 TFLOPS | Memory BW |
|----------|------|-------------|-----------|
| Jetson AGX Orin 64GB | Edge | ~138 (FP16) | 204 GB/s |
| A800 80GB | Server | 312 | 2039 GB/s |
| A100 40/80GB | Server | 312 | 1555/2039 GB/s |
| H100 SXM | Server | 989 | 3350 GB/s |
| B100 | Server | 1750 | 8000 GB/s |
| RTX 4090 | Desktop | 165 (FP16) | 1008 GB/s |
| RTX 5090 | Desktop | 209 | 1792 GB/s |
| Jetson AGX Thor | Edge | 800 (FP8) | 270 GB/s |

Full list in GenZ's `system_configs.py` (25+ platforms).

## Acknowledgments

- [NVIDIA VLA-Perf](https://arxiv.org/abs/2602.18397) — the foundation for VLA inference latency modeling
- [GenZ LLM Analyzer](https://github.com/abhibambhaniya/GenZ-LLM-Analyzer) — roofline-based analytical performance modeling engine
- [SigLIP2](https://arxiv.org/abs/2502.14786), [Qwen2.5](https://arxiv.org/abs/2412.15115) — vision encoder and language backbone model families

## License

Apache 2.0
