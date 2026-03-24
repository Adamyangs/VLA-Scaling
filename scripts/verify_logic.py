#!/usr/bin/env python3
"""Comprehensive verification of simulation logic for all 6 action topologies."""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)

from vla_bench.engine import VLAPerfEngine
from vla_bench.configs import VLAConfig, get_config_by_id

engine = VLAPerfEngine(batch_size=1, bits="bf16")
HW = "A100_80GB"
passed = 0
failed = 0


def check(name, condition, msg=""):
    global passed, failed
    if condition:
        print(f"  OK: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} — {msg}")
        failed += 1


def make(vk="V-M", lk="L-M", ak="Cascade-M", chunk=None, steps=None):
    return VLAConfig(0, "T", "T", vision_key=vk, language_key=lk,
                     action_key=ak, chunk_size=chunk, denoising_steps=steps)


print("=" * 80)
print("VLA ARCHITECTURE SIMULATION LOGIC VERIFICATION")
print("=" * 80)

# ---- Test 1: Vision encoder token counts ----
print("\n[Test 1] Vision Encoder Token Counts")
for vk, expected in [("V-S", 256), ("V-M", 576), ("V-L", 256)]:
    cfg = make(vk=vk)
    check(f"{vk} tokens={cfg.vision_tokens}", cfg.vision_tokens == expected,
          f"expected {expected}, got {cfg.vision_tokens}")

# ---- Test 2: VLM sequence length ----
print("\n[Test 2] VLM Sequence Length = vision_tokens + language_tokens")
for vk, lk in [("V-S", "L-S"), ("V-M", "L-M"), ("V-L", "L-L")]:
    cfg = make(vk=vk, lk=lk)
    expected = cfg.vision_tokens + cfg.language_tokens
    check(f"{vk}+{lk}: {cfg.vlm_sequence_length}=={expected}",
          cfg.vlm_sequence_length == expected)

# ---- Test 3: Cascade Denoise — DiT model, linear in steps ----
print("\n[Test 3] Cascade Denoise: separate DiT, latency linear in steps")
r10 = engine.evaluate_action(make(ak="Cascade-M"), HW)
r20 = engine.evaluate_action(make(ak="Cascade-M", steps=20), HW)
ratio = r20.time_ms / r10.time_ms
# Ratio is <2.0x because KV projection is a fixed one-time cost
check(f"steps 10->20: {r10.time_ms:.2f}->{r20.time_ms:.2f}ms, ratio={ratio:.2f}x (<2.0x due to KV proj)",
      1.7 < ratio < 2.0, f"expected ~1.8x, got {ratio:.2f}x")

# Size scaling: S < M < L (different DiT)
rs = engine.evaluate_action(make(ak="Cascade-S"), HW)
rl = engine.evaluate_action(make(ak="Cascade-L"), HW)
check(f"S({rs.time_ms:.2f}) < M({r10.time_ms:.2f}) < L({rl.time_ms:.2f})",
      rs.time_ms < r10.time_ms < rl.time_ms)

# ---- Test 4: SharedAttn Denoise — uses action expert (pi0-style) ----
# Pi0: action tokens go through action expert's layers (separate FFN),
# cross-attend to VLM KV cache. Expert model determines action latency.
print("\n[Test 4] SharedAttn Denoise: uses action expert (pi0-style)")
sa_s = engine.evaluate_action(make(ak="SharedAttn-S"), HW)
sa_m = engine.evaluate_action(make(ak="SharedAttn-M"), HW)
sa_l = engine.evaluate_action(make(ak="SharedAttn-L"), HW)
check(f"S({sa_s.time_ms:.2f}) < M({sa_m.time_ms:.2f}) < L({sa_l.time_ms:.2f})",
      sa_s.time_ms < sa_m.time_ms < sa_l.time_ms)

# SharedAttn-M < Cascade-M (SharedAttn saves KV projection cost)
cascade_m_time = r10.time_ms  # Cascade-M includes KV projection
check(f"SharedAttn-M({sa_m.time_ms:.2f}) < Cascade-M({cascade_m_time:.2f}) (no KV proj)",
      sa_m.time_ms < cascade_m_time)

# Action latency varies modestly with VLM seq len (affects cross-attn KV size)
sa_small = engine.evaluate_action(make(lk="L-S", ak="SharedAttn-M"), HW)
sa_large = engine.evaluate_action(make(lk="L-L", ak="SharedAttn-M"), HW)
check(f"L-S({sa_small.time_ms:.2f}) ~ L-L({sa_large.time_ms:.2f}), ratio<2x",
      sa_large.time_ms / sa_small.time_ms < 2.0)

# ---- Test 5: CrossAttn Denoise — same as Cascade (both have KV projection) ----
print("\n[Test 5] CrossAttn Denoise: same as Cascade (both pay KV projection)")
ca = engine.evaluate_action(make(ak="CrossAttn-M"), HW)
check(f"CrossAttn-M({ca.time_ms:.4f}) == Cascade-M({r10.time_ms:.4f})",
      abs(ca.time_ms - r10.time_ms) < 0.01)
check(f"CrossAttn-M({ca.time_ms:.2f}) > SharedAttn-M({sa_m.time_ms:.2f})",
      ca.time_ms > sa_m.time_ms)

# ---- Test 6: AR-Naive — dof * chunk tokens, linear scaling ----
print("\n[Test 6] AR-Naive: tokens = dof * chunk, linear in both")
ar1 = engine.evaluate_action(make(ak="AR-Naive", chunk=1), HW)
ar10 = engine.evaluate_action(make(ak="AR-Naive", chunk=10), HW)
ratio = ar10.time_ms / ar1.time_ms
check(f"chunk 1->10: {ar1.time_ms:.2f}->{ar10.time_ms:.2f}ms, ratio={ratio:.1f}x",
      abs(ratio - 10.0) < 0.2, f"expected 10x, got {ratio:.1f}x")

# Verify token count: chunk=1, dof=7 → 7 tokens
ar2 = engine.evaluate_action(make(ak="AR-Naive", chunk=2), HW)
ratio2 = ar2.time_ms / ar1.time_ms
check(f"chunk 1->2: ratio={ratio2:.2f}x (expected 2.0x)", abs(ratio2 - 2.0) < 0.1)

# AR scales with VLM size
ar_s = engine.evaluate_action(make(lk="L-S", ak="AR-Naive", chunk=10), HW)
ar_l = engine.evaluate_action(make(lk="L-L", ak="AR-Naive", chunk=10), HW)
check(f"VLM L-S({ar_s.time_ms:.2f}) < L-L({ar_l.time_ms:.2f})",
      ar_l.time_ms > ar_s.time_ms * 1.5)

# ---- Test 7: AR-FAST — ~5x compression vs Naive ----
print("\n[Test 7] AR-FAST: ~5x fewer tokens than AR-Naive")
fast10 = engine.evaluate_action(make(ak="AR-FAST", chunk=10), HW)
ratio = ar10.time_ms / fast10.time_ms
check(f"Naive({ar10.time_ms:.2f}) / FAST({fast10.time_ms:.2f}) = {ratio:.1f}x",
      abs(ratio - 5.0) < 0.2, f"expected ~5x, got {ratio:.1f}x")

# Verify exact token count: chunk=10, dof=7 → naive=70, FAST=70/5=14
fast1 = engine.evaluate_action(make(ak="AR-FAST", chunk=1), HW)
fast_ratio = fast10.time_ms / fast1.time_ms
# chunk=1: naive=7, FAST=max(1, 7//5)=1 token; chunk=10: FAST=14 tokens
check(f"FAST chunk 1->10 ratio={fast_ratio:.1f}x (expected 14x)",
      abs(fast_ratio - 14.0) < 1.0, f"got {fast_ratio:.1f}x")

# ---- Test 8: Regression — single MLP pass, chunk-invariant ----
print("\n[Test 8] Regression: single pass, near-zero, chunk-invariant")
reg = engine.evaluate_action(make(ak="Regress-M"), HW)
check(f"Regress-M: {reg.time_ms:.4f}ms < 0.5ms", reg.time_ms < 0.5)

reg50 = engine.evaluate_action(make(ak="Regress-M", chunk=50), HW)
check(f"chunk=10({reg.time_ms:.4f}) == chunk=50({reg50.time_ms:.4f})",
      abs(reg.time_ms - reg50.time_ms) < 0.001)

# Size has minimal effect
reg_s = engine.evaluate_action(make(ak="Regress-S"), HW)
reg_l = engine.evaluate_action(make(ak="Regress-L"), HW)
check(f"All < 0.1ms: S={reg_s.time_ms:.4f}, M={reg.time_ms:.4f}, L={reg_l.time_ms:.4f}",
      max(reg_s.time_ms, reg.time_ms, reg_l.time_ms) < 0.1)

# Regression does NOT scale with VLM (uses separate MLP)
reg_s_vlm = engine.evaluate_action(make(lk="L-S", ak="Regress-M"), HW)
reg_l_vlm = engine.evaluate_action(make(lk="L-L", ak="Regress-M"), HW)
check(f"VLM-invariant: L-S({reg_s_vlm.time_ms:.4f}) == L-L({reg_l_vlm.time_ms:.4f})",
      abs(reg_s_vlm.time_ms - reg_l_vlm.time_ms) < 0.001)

# ---- Test 9: E2E = Vision + VLM + Action ----
print("\n[Test 9] E2E = Vision + VLM + Action")
for cid in [1, 17, 32, 44, 55]:
    try:
        cfg = get_config_by_id(cid)
    except ValueError:
        continue
    r = engine.evaluate_e2e(cfg, HW)
    if r is None:
        continue
    expected = r["vision_time_ms"] + r["vlm_time_ms"] + r["action_time_ms"]
    check(f"#{cid} E2E={r['e2e_time_ms']:.4f} == V+L+A={expected:.4f}",
          abs(r["e2e_time_ms"] - expected) < 0.001)
    check(f"#{cid} Hz={r['e2e_hz']:.1f} == 1000/{r['e2e_time_ms']:.2f}",
          abs(r["e2e_hz"] - 1000 / r["e2e_time_ms"]) < 0.01)

# ---- Test 10: Memory fit with model_name=None (AR types) ----
print("\n[Test 10] Memory Fit Check (model_name=None)")
for ak in ["AR-Naive", "AR-FAST"]:
    cfg = make(ak=ak)
    fits, reason = engine.check_memory_fit(cfg, HW)
    check(f"{ak} on A100: {reason}", fits)

# ---- Summary ----
print("\n" + "=" * 80)
total = passed + failed
print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED — review output above")
    sys.exit(1)
