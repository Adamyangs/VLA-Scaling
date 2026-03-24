# VLA-Perf++ Design Document

## 1. Overview

VLA-Perf++ is an analytical roofline benchmark for Vision-Language-Action (VLA) models.
It evaluates **64 VLA configurations** across 3 component axes:

- **Vision Encoder (V)**: SigLIP2-B (86M), SigLIP2-L (307M), SigLIP2-So (400M)
- **Language Backbone (L)**: Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B
- **Action Head (A)**: 4 architectures × 3 sizes = FM/Diff/AR/MLP

The project extends NVIDIA's VLA-Perf and GenZ LLM Analyzer to answer 7 research questions
about VLA architecture design, component scaling, and deployment trade-offs.

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  VLA-Scaling (this project)                                        │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │  configs.py   │──>│  engine.py   │──>│  results/*.csv       │    │
│  │  64 VLAConfig │   │ VLAPerfEngine│   │  per-component data  │    │
│  └──────────────┘   └──────┬───────┘   └──────────────────────┘    │
│                            │                                        │
│  ┌──────────────┐          │           ┌──────────────────────┐    │
│  │  network.py   │          │           │  scripts/plot/*.py   │    │
│  │  8 scenarios  │          │           │  5 plot scripts      │    │
│  └──────────────┘          │           └──────────────────────┘    │
├────────────────────────────┼────────────────────────────────────────┤
│  VLA-Perf / GenZ (NVIDIA)  │                                        │
│                            ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │  GenZ LLM Analyzer (installed via pip install -e .)       │      │
│  │                                                           │      │
│  │  prefill_moddeling()    ── Vision Encoder, VLM Backbone   │      │
│  │  decode_moddeling()     ── AR action head                 │      │
│  │  parallel_decode_modeling() ── FM/Diff action heads       │      │
│  │                                                           │      │
│  │  vla_models.py   ── Model configs (architecture specs)    │      │
│  │  system_configs.py ── Hardware specs (FLOPS, BW, memory)  │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. VLA Inference Pipeline Modeling

A VLA model processes visual input and language instructions to produce robot actions.
The inference pipeline is modeled as 3 sequential stages:

```
Camera Image(s)  ─→  [Vision Encoder]  ─→  Visual Tokens
                         (prefill)            │
Language Prompt  ─→  ─────────────────────────┤
                                              ▼
                     [Language Backbone]  ─→  KV Cache + Hidden States
                         (prefill)            │
                                              ▼
                     [Action Head]       ─→  Action Chunk
                     (varies by type)      (joint positions × T steps)
```

### 3.1 Action Head Modeling

The 4 action head types map to different GenZ inference modes:

| Type | GenZ API | Inference Pattern | Latency Formula |
|------|----------|-------------------|-----------------|
| **FM** (Flow Matching) | `parallel_decode_modeling()` | All action tokens decoded in parallel per step | `per_step × N_steps` |
| **Diff** (Diffusion) | `parallel_decode_modeling()` | Same as FM (different training, same arch) | `per_step × N_steps` |
| **AR** (Autoregressive) | `decode_moddeling()` | One action token at a time, sequential | `per_token × chunk_size` |
| **MLP** (Regression) | `prefill_moddeling()` | Single forward pass through shallow model | `1 × prefill_time` |

**Key properties:**
- FM/Diff: **constant** in chunk_size (parallel decode), **linear** in denoising steps
- AR: **linear** in chunk_size (sequential generation), no denoising steps
- MLP: **negligible** latency (<0.5ms on any GPU), no iteration

### 3.2 Roofline Analysis

Each pipeline stage is evaluated through GenZ's roofline model:

```
                                  ┌──── Compute Bound ────┐
                                  │  (large batch, large   │
Performance                       │   sequence, compute-   │
(GFLOPS)                          │   intensive ops)       │
    ▲           ╱─────────────────┤                        │
    │         ╱                   └────────────────────────┘
    │       ╱ ← Ridge Point
    │     ╱
    │   ╱  ← Memory Bound
    │ ╱     (small batch, KV cache reads, weight-dominated)
    └──────────────────────────────────────────▶
           Operational Intensity (FLOP/Byte)
```

For each component, GenZ computes per-layer:
1. **Compute time** = Total FLOPs / Hardware peak FLOPS
2. **Memory time** = Data moved / Memory bandwidth
3. **Communication time** = Collective data / Interconnect bandwidth (for TP/PP)
4. **Layer time** = max(compute, memory, communication)
5. **Boundness** = which of the 3 dominates

## 4. Configuration Registry Design

### 4.1 The 64-Config Matrix

Organized into 3 phases and 13 groups, each answering a specific research question:

```
Phase P0 (40 configs) → Q1-Q4
├── Group A (9):  3×3 V×L grid × FM-M         → Q2 (optimal allocation), Q3 (bottleneck)
├── Group B (9):  V-Scaling × {Diff,AR,MLP}    → Q1 (V scaling efficiency)
├── Group C (6):  L-Scaling × {Diff,AR,MLP}    → Q1 (L scaling efficiency)
├── Group D (6):  A-Scaling × {FM,Diff,MLP}    → Q4 (action architecture vs size)
├── Group E (6):  Corner VLMs × {Diff,AR,MLP}  → Q4 (generalization)
└── Group F (4):  FM {S,L} × Corner VLMs       → Q4 (FM size scaling)

Phase P1 (12 configs) → Q5
├── Group G (4):  FM-M chunk sweep {1,5,25,50}
├── Group H (3):  FM-M steps sweep {5,25,50}
├── Group I (3):  AR chunk sweep {1,5,25}
└── Group J (2):  MLP-S on corner VLMs

Phase P2 (12 configs) → Q5-Q7 supplementary
├── Group K (4):  Chunk generalization on VLM-1/9
├── Group L (4):  Steps generalization on VLM-1/9
└── Group M (4):  Diff S/L on corner VLMs
```

### 4.2 VLM Matrix

The 9 VLM backbones form a 3×3 grid used across multiple experiments:

```
              Qwen2.5-0.5B   Qwen2.5-1.5B   Qwen2.5-3B
             ┌─────────────┬──────────────┬─────────────┐
SigLIP2-B    │  VLM-① 0.6B │  VLM-② 1.6B │  VLM-③ 3.1B│
             ├─────────────┼──────────────┼─────────────┤
SigLIP2-L    │  VLM-④ 0.8B │  VLM-⑤ 1.8B │  VLM-⑥ 3.3B│
             │             │  ★ baseline  │             │
             ├─────────────┼──────────────┼─────────────┤
SigLIP2-So   │  VLM-⑦ 0.9B │  VLM-⑧ 1.9B │  VLM-⑨ 3.4B│
             └─────────────┴──────────────┴─────────────┘
```

## 5. Hardware and Deployment

### 5.1 Target Platforms

| Platform | Type | Memory | Compute (BF16) | Mem BW |
|----------|------|--------|-----------------|--------|
| Jetson AGX Orin 64GB | Edge | 64GB shared | ~138 TFLOPS | 204 GB/s |
| A800 80GB | Server | 80GB HBM2e | 312 TFLOPS | 2039 GB/s |

### 5.2 Deployment Scenarios

```
Scenario 1: On-Device (all on Orin)
  Robot ──[V]──[L]──[A]──→ Actions

Scenario 2: Full Offload (all on server)
  Robot ──image──→ [WiFi/Eth] ──→ Server ──[V]──[L]──[A]──→ [WiFi/Eth] ──action──→ Robot

Scenario 3: Split Inference (vision on edge, VLM+action on server)
  Robot ──[V]──features──→ [WiFi/Eth] ──→ Server ──[L]──[A]──→ [WiFi/Eth] ──action──→ Robot
```

Network overhead is modeled for WiFi 5/6/7, 1G/10G/25G/100G Ethernet, and cloud links.

### 5.3 Parallelism

For multi-device setups, GenZ explores all valid (TP, PP) combinations:
- **Tensor Parallel (TP)**: Splits layers across devices (requires NVLink/ICN)
- **Pipeline Parallel (PP)**: Distributes layers sequentially across devices
- Only TP > 1 is allowed when the system has ICN > 0 (interconnect bandwidth)
- The engine automatically selects the best (TP, PP) combination per component

## 6. Model Configuration Design

### 6.1 Why `vocab_size=0` for VLA Backbones

Standard LLM configs include embedding and LM head parameters (`vocab_size × hidden_size`).
VLA models use the LLM as a backbone:
- **Input**: Visual tokens (from vision encoder projector) + language tokens
- **Output**: Hidden states → passed to action head (not to LM head)

Setting `vocab_size=0` removes the LM head from latency modeling. Without this,
VLM latency is overestimated by 17-24% due to a phantom output projection.

### 6.2 Action Head Architecture Specs

| Config | Layers | Hidden | Intermediate | Heads | KV Heads | ~Params |
|--------|--------|--------|--------------|-------|----------|---------|
| FM/Diff-S | 8 | 640 | 2560 | 8 | 2 | 46M |
| FM/Diff-M | 14 | 1024 | 4096 | 8 | 2 | 209M |
| FM/Diff-L | 19 | 1280 | 5120 | 16 | 4 | 452M |
| MLP-S | 2 enc | 512 | 2048 | 8 | 8 | ~10M |
| MLP-M | 3 enc | 768 | 3072 | 12 | 12 | ~30M |
| MLP-L | 4 enc | 1024 | 4096 | 16 | 16 | ~80M |
| AR | (reuses LLM backbone) | — | — | — | — | 0 extra |

FM and Diff share identical transformer architectures. The difference is in training
(continuous flow matching vs discrete DDPM schedule), not inference computation.

## 7. Output Schema

The benchmark produces a CSV with one row per (config, hardware) combination.
Key columns:

```
config_id, config_name, group, phase,
vision_key, language_key, action_key, action_type,
hardware, num_devices, bits, batch_size,
chunk_size, denoising_steps, num_frames, vision_tokens, vlm_seq_len,

# Per-component breakdown (ms)
vision_time_ms, vlm_time_ms, action_time_ms, e2e_time_ms,

# Roofline classification
vision_boundness, vlm_boundness, action_boundness,
vision_op_intensity, vlm_op_intensity, action_op_intensity,

# Memory footprint (MB)
vision_weights_mb, vlm_weights_mb, action_weights_mb, vlm_kv_cache_mb, total_memory_mb,

# Derived
e2e_hz,    # 1000 / e2e_time_ms
ttfa_ms,   # Time to first action = vision + vlm
```

## 8. Known Limitations

1. **Analytical, not measured**: Latency is estimated via roofline model, not measured on real hardware.
   Actual latency depends on kernel implementations, memory fragmentation, scheduling overhead, etc.
   GenZ typically achieves ~80% accuracy vs real measurements (MAPE ~20%).

2. **Projector not modeled**: The vision-to-LLM projector (typically a 2-layer MLP mapping
   vision features to LLM hidden space) is not explicitly modeled. Its latency is negligible
   (<0.1ms) compared to vision/VLM/action components.

3. **MLP heads modeled as transformers**: GenZ's roofline model is designed for transformer
   architectures. MLP action heads are approximated as shallow encoder-only transformers,
   which slightly overestimates their latency (adds unnecessary attention computation).

4. **No memory fragmentation**: The memory fit check uses ideal packing. Real systems may
   have fragmentation, activation memory, and optimizer states that reduce available memory.

5. **Single-batch focus**: The benchmark defaults to batch_size=1 (real-time robotics).
   Throughput-oriented scenarios (batch>1) are supported but not the primary focus.
