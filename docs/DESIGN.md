# VLA-Perf++ Design Document

## 1. Overview

VLA-Perf++ is an analytical roofline benchmark for Vision-Language-Action models.
It evaluates **80 VLA configurations** across 3 component axes:

- **Vision Encoder (V)**: SigLIP2-B (86M), SigLIP2-L (307M), SigLIP2-So (400M)
- **Language Backbone (L)**: Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B
- **Action Head (A)**: 6 topologies — Cascade / SharedAttn / CrossAttn / AR-Naive / AR-FAST / Regression

Built on NVIDIA VLA-Perf and GenZ LLM Analyzer.

## 2. System Architecture

```
VLA-Scaling
├── vla_bench/
│   ├── configs.py ─── 80 VLAConfig (3 phases, 11 groups)
│   │                      │
│   ├── engine.py ─── VLAPerfEngine
│   │                   ├── evaluate_vision()   → prefill_moddeling()
│   │                   ├── evaluate_vlm()      → prefill_moddeling()
│   │                   ├── evaluate_action()   → depends on topology:
│   │                   │     cascade_denoise    → parallel_decode × N steps
│   │                   │     shared_attn_denoise→ parallel_decode × N (on VLM)
│   │                   │     cross_attn_denoise → parallel_decode × N steps
│   │                   │     ar_naive           → decode × (dof × chunk)
│   │                   │     ar_fast            → decode × (dof × chunk / 5)
│   │                   │     regression         → prefill × 1 token
│   │                   └── evaluate_e2e()      → V + L + A
│   │
│   └── network.py ─── 8 deployment scenarios
│
├── scripts/
│   ├── run_benchmark.py ─── CLI entry point
│   ├── run_scaling.py ─── Per-question experiments (Q1-Q7)
│   └── plot/
│       └── plot_comparison.py ─── 6 controlled-variable figures
│
└── GenZ (../vla-perf/genz, installed via pip install -e .)
    ├── vla_models.py ── Model configs (Qwen2.5, denoise experts, MLP heads)
    ├── system_configs.py ── Hardware specs (25+ platforms)
    └── LLM_inference/ ── prefill / decode / parallel_decode modeling
```

## 3. VLA Inference Pipeline

```
Camera Image(s) ──→ [Vision Encoder] ──→ Visual Tokens
                        (prefill)           │
Language Prompt ──→ ───────────────────────┤
                                            ▼
                    [Language Backbone] ──→ KV Cache + Hidden States
                        (prefill)           │
                                            ▼
                    [Action Head]       ──→ Action Chunk
                    (topology-dependent)    (dof × chunk_size values)
```

## 4. The 6 Action Head Topologies

### 4.1 Classification Rationale

Previously, VLA action heads were classified by generation strategy (Flow Matching vs Diffusion vs AR vs MLP). This is wrong for latency analysis because:

- **FM and Diffusion have identical latency** — same DiT architecture, only training loss differs
- **The connection topology is what determines latency** — how the action expert receives VLM information

Our classification is based on **how the action head connects to the VLM backbone**:

### 4.2 Topology Details

#### Type 1: Cascade Denoise
```
VLM ──prefill──→ [KV Cache] ──cross-attn──→ [DiT] ──×N steps──→ Actions
                              (sequential)
```
- **Representatives:** GR00T N1, CogACT, DexVLA
- **GenZ modeling:** `parallel_decode_modeling(DiT, input=VLM_seq, output=chunk)` × N steps
- **Latency:** VLM prefill + DiT_per_step × N_steps
- **Key property:** DiT size directly affects action latency

#### Type 2: SharedAttn Denoise
```
VLM + Action tokens ──shared self-attn──→ [VLM layers] ──×N steps──→ Actions
                      (KV cache reused)
```
- **Representatives:** pi0, ForceVLA, OneTwoVLA
- **GenZ modeling:** `parallel_decode_modeling(VLM, input=VLM_seq, output=chunk)` × N steps
- **Latency:** VLM prefill + VLM_per_step × N_steps
- **Key property:** Action latency scales with VLM size, NOT with expert config size. The expert model config is ignored — we use the VLM backbone itself for denoising.

#### Type 3: CrossAttn Denoise
```
VLM ──prefill──→ [Features] ──cross-attn──→ [Lightweight DiT] ──×N steps──→ Actions
                  (cached)      (interleaved SA+CA blocks)
```
- **Representatives:** SmolVLA, GR-3
- **GenZ modeling:** same as Cascade (analytically equivalent at the roofline level)
- **Latency:** VLM prefill + DiT_per_step × N_steps
- **Key property:** Same analytical cost as Cascade. Real-world difference is in kernel fusion and the interleaved SA+CA block pattern.

#### Type 4a: AR-Naive
```
VLM ──prefill──→ [KV Cache] ──decode──→ token₁ ──→ token₂ ──→ ... ──→ tokenₙ
                              (one by one, n = dof × chunk_size)
```
- **Representatives:** OpenVLA, RT-2
- **GenZ modeling:** `decode_moddeling(VLM)` × (action_dof × chunk_size)
- **Latency:** VLM prefill + per_token_decode × dof × chunk_size
- **Key property:** Linear in BOTH dof and chunk_size. With 7 DoF and chunk=10, generates 70 tokens.

#### Type 4b: AR-FAST
```
Action chunk ──DCT──→ freq coefficients ──BPE──→ compressed tokens (÷5)
VLM ──prefill──→ [KV Cache] ──decode──→ token₁ ──→ ... ──→ tokenₖ  (k ≈ n/5)
```
- **Representatives:** pi0-FAST
- **GenZ modeling:** `decode_moddeling(VLM)` × ceil(action_dof × chunk_size / compression_ratio)
- **Latency:** VLM prefill + per_token_decode × (dof × chunk / 5)
- **Key property:** ~5x fewer tokens than naive AR due to DCT frequency-domain compression.

#### Type 5: Direct Regression
```
VLM ──prefill──→ hidden[<ACT>] ──MLP──→ continuous actions (single pass)
```
- **Representatives:** OpenVLA-OFT, BridgeVLA, VOTE
- **GenZ modeling:** `prefill_moddeling(MLP_head, input=1_token)`
- **Latency:** VLM prefill + MLP_forward (negligible)
- **Key property:** Fastest possible. Action head contributes <1% of E2E latency.

### 4.3 Latency Comparison Summary

```
                    Slowest ←──────────────────────────────→ Fastest
On Edge (Orin):
  AR-Naive ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  864ms
  AR-FAST  ▓▓▓▓▓▓▓▓                                192ms
  SharedAtn▓▓▓▓▓▓▓                                 153ms
  Cascade-L▓▓▓                                      71ms
  Cascade-M▓▓                                       46ms
  Cascade-S▓▓                                       29ms
  Regression▓                                       25ms

On Server (A800):
  AR-Naive ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓        87ms
  AR-FAST  ▓▓▓▓▓▓                                  20ms
  SharedAtn▓▓▓▓▓                                   16ms
  Cascade-L▓▓                                       8ms
  Cascade-M▓▓                                       5ms
  Cascade-S▓                                        4ms
  Regression▓                                       3ms
```

## 5. Configuration Registry

### 5.1 VLM Matrix (3×3)

```
              Qwen2.5-0.5B   Qwen2.5-1.5B   Qwen2.5-3B
SigLIP2-B     VLM-1 (0.6B)   VLM-2 (1.6B)   VLM-3 (3.1B)
SigLIP2-L     VLM-4 (0.8B)   VLM-5 (1.8B)*  VLM-6 (3.3B)
SigLIP2-So    VLM-7 (0.9B)   VLM-8 (1.9B)   VLM-9 (3.4B)

* VLM-5 is the baseline for controlled experiments
```

### 5.2 Experiment Groups (80 configs)

```
Phase P0 (55 configs) → Q1-Q4
├── A (9):  3×3 V×L grid × Cascade-M
├── B (15): V-Scaling × {SharedAttn, CrossAttn, AR-Naive, AR-FAST, Regress}
├── C (10): L-Scaling × {SharedAttn, CrossAttn, AR-Naive, AR-FAST, Regress}
├── D (9):  Size scaling S/M/L × {Cascade, SharedAttn, Regress}
└── E (12): Corner VLMs × all 6 topologies

Phase P1 (13 configs) → Q5
├── G (4):  Cascade chunk sweep {1, 5, 25, 50}
├── H (3):  Cascade steps sweep {5, 25, 50}
└── I (6):  AR-Naive vs AR-FAST chunk sweep {1, 10, 50}

Phase P2 (12 configs) → Q5-Q7
├── K (4):  Chunk generalization on VLM-1/9
├── L (4):  Steps generalization on VLM-1/9
└── M (4):  CrossAttn S/L on corner VLMs
```

## 6. Model Configuration Notes

### vocab_size=0 for VLA Backbones

VLA models use the LLM as a backbone without the LM head (action heads handle output). Setting `vocab_size=0` removes the output projection from latency modeling. Without this, VLM latency is overestimated by 17-24%.

### Model Name Resolution

Language model names MUST use `vla/` prefix (e.g., `vla/qwen2.5-1.5b`) to avoid resolving to GenZ's pre-existing configs in `alibaba.py` which include vocab embeddings.

### Denoise Expert Unification

FM and Diffusion action experts are defined as a single `denoise-expert-{s,m,l}` config set. Legacy aliases (`fm-action-expert-*`, `diff-action-expert-*`) point to the same configs for backward compatibility.

## 7. Output Schema

One row per (config, hardware) combination:

```
# Identifiers
config_id, config_name, group, phase, vision_key, language_key, action_key, action_type

# Hardware
hardware, num_devices, bits, batch_size

# Parameters
chunk_size, denoising_steps, num_frames, vision_tokens, vlm_seq_len

# Per-component latency (ms)
vision_time_ms, vlm_time_ms, action_time_ms, e2e_time_ms

# Roofline
vision_boundness, vlm_boundness, action_boundness  (Comp/Mem/Comm)
vision_op_intensity, vlm_op_intensity, action_op_intensity  (FLOP/Byte)

# Memory
vision_weights_mb, vlm_weights_mb, action_weights_mb, vlm_kv_cache_mb, total_memory_mb

# Derived
e2e_hz (1000/e2e_time_ms), ttfa_ms (vision+vlm)
```

## 8. Known Limitations

1. **Analytical model, not measured.** GenZ roofline accuracy is ~80% (MAPE ~20%). Real latency depends on kernel implementations and scheduling.
2. **Cascade = CrossAttn analytically.** Both use the same DiT with cross-attention. Real-world differences from interleaved block patterns and kernel fusion are not captured.
3. **SharedAttn ignores expert config size.** By design — it uses the VLM backbone, not a separate expert. The S/M/L suffix is kept for API consistency but has no effect.
4. **AR-FAST compression ratio is approximate.** Fixed at 5x; actual ratio depends on action smoothness and BPE vocabulary.
5. **Projector not modeled.** Vision-to-LLM projector (<0.1ms) is omitted.
6. **Single-batch focus.** Default batch_size=1 for real-time robotics.
