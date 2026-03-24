# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLA-Perf++ (VLA-Scaling) is a roofline benchmark for Vision-Language-Action models. It evaluates inference latency for 64 VLA configurations across 3 vision encoders × 3 language backbones × 4 action head architectures, on edge and server hardware.

Built on top of NVIDIA's VLA-Perf / GenZ LLM Analyzer.

## Setup

```bash
# Install GenZ backend (required)
cd ../vla-perf/genz
pip install -e .

# Install dependencies
cd ../../VLA-Scaling
pip install -r requirements.txt
```

## Commands

```bash
# Run full benchmark (64 configs × 2 systems)
python scripts/run_benchmark.py

# Run specific phase/group/configs
python scripts/run_benchmark.py --phase P0
python scripts/run_benchmark.py --group A
python scripts/run_benchmark.py --configs 1 5 9

# Run on specific hardware
python scripts/run_benchmark.py --systems H100 RTX_4090

# Run scaling experiments
python scripts/run_scaling.py --experiment q1  # Component scaling
python scripts/run_scaling.py --experiment all

# Generate plots
python scripts/plot/plot_breakdown.py --system Jetson_AGX_Orin_64GB
python scripts/plot/plot_heatmap.py --system A800_80GB
python scripts/plot/plot_scaling.py --system A800_80GB
python scripts/plot/plot_roofline.py --system A800_80GB
python scripts/plot/plot_pareto.py --system A800_80GB
```

## Architecture

### vla_bench/ — Core package
- **configs.py** — 64 VLA configurations organized in 3 phases (P0/P1/P2) and 13 groups (A-M). Each VLAConfig specifies vision encoder, language backbone, and action head. Model components are registered in GenZ's vla_models.py.
- **engine.py** — VLAPerfEngine wraps GenZ roofline analysis for 4 action types:
  - Flow Matching (FM): `parallel_decode_modeling` × N denoising steps
  - Diffusion (Diff): same as FM architecturally, different training
  - Autoregressive (AR): `decode_moddeling` × chunk_size (reuses LLM backbone)
  - MLP: `prefill_moddeling` with 1 token through shallow model
- **network.py** — Network latency for deployment scenarios (on-device, full offload, split inference)

### Model configs (in ../vla-perf/genz/)
- Vision: SigLIP2-B (86M), SigLIP2-L (307M), SigLIP2-So (400M) — `vla_models.py`
- Language: Qwen2.5-0.5B/1.5B/3B — `vla_models.py`
- Action: FM-S/M/L, Diff-S/M/L, MLP-S/M/L — `vla_models.py`
- Hardware: A800_80GB, Jetson_AGX_Orin_64GB + existing — `system_configs.py`

### Experiment groups (from paper's 7 research questions)
- **A**: 3×3 V×L grid (Q2, Q3)
- **B**: V-Scaling × Diff/AR/MLP (Q1)
- **C**: L-Scaling × Diff/AR/MLP (Q1)
- **D**: A-Scaling on VLM-5 (Q4)
- **E/F**: Corner VLM generalization (Q4)
- **G/H/I**: Chunk size and denoising steps sweep (Q5)
- **J**: MLP-S supplementary (Q4)
- **K/L/M**: P2 generalization experiments (Q5-Q7)

## Key Design Decisions

- Action type determines modeling approach: FM/Diff use `parallel_decode_modeling` (all action tokens decoded simultaneously per step), AR uses `decode_moddeling` (one token at a time, multiplied by chunk_size)
- FM and Diff have identical latency characteristics (same transformer arch, different training objectives)
- AR reuses the LLM backbone — no separate action model needed
- MLP heads are modeled as shallow encoder-only transformers
- batch_size=1 is default (real-time robotics scenario)
- VLA backbone configs use `vocab_size=0` — the LM head is not used (action heads handle output). Without this, VLM latency is overestimated by ~20%
- Language model names MUST use `vla/` prefix (e.g., `vla/qwen2.5-1.5b`) to avoid resolving to GenZ's pre-existing configs in `alibaba.py` which include vocab embeddings

## Design Documentation

See [docs/DESIGN.md](docs/DESIGN.md) for detailed architecture, pipeline modeling, and configuration design.
