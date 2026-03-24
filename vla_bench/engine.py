"""
VLA Performance Evaluation Engine

Wraps GenZ's roofline analysis to evaluate all 4 action head types:
  - Flow Matching (FM): parallel_decode × N denoising steps
  - Diffusion (Diff): parallel_decode × N denoising steps
  - Autoregressive (AR): decode, sequential token generation
  - MLP: prefill with 1 token through shallow model

Outputs per-component latency breakdown:
  Vision Encoder | Projector | Language Backbone | Action Head
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

# Add vla-perf paths for GenZ imports
_VLA_PERF_ROOT = Path(__file__).resolve().parent.parent.parent / "vla-perf"
_GENZ_ROOT = _VLA_PERF_ROOT / "genz"
_VLA_PERF_DIR = _VLA_PERF_ROOT / "vla-perf"
for p in [str(_GENZ_ROOT), str(_VLA_PERF_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from GenZ import prefill_moddeling, decode_moddeling, parallel_decode_modeling
from GenZ.Models import get_configs
from GenZ.unit import Unit
from Systems.system_configs import system_configs

from perf_utils import (
    evaluate_boundness,
    get_best_precision_for_system,
    get_parallelism,
    calculate_transformer_params,
)

from .configs import VLAConfig

logger = logging.getLogger(__name__)


# Extended result columns with component-level breakdown
BENCH_COLUMNS = [
    # Config identifiers
    "config_id",
    "config_name",
    "group",
    "phase",
    # Component names
    "vision_key",
    "language_key",
    "action_key",
    "action_type",
    # Hardware
    "hardware",
    "num_devices",
    "bits",
    # Inference parameters
    "batch_size",
    "chunk_size",
    "denoising_steps",
    "num_frames",
    "vision_tokens",
    "vlm_seq_len",
    # Per-component latency (ms)
    "vision_time_ms",
    "vlm_time_ms",
    "action_time_ms",
    "e2e_time_ms",
    # Per-component boundness
    "vision_boundness",
    "vlm_boundness",
    "action_boundness",
    # Per-component op intensity
    "vision_op_intensity",
    "vlm_op_intensity",
    "action_op_intensity",
    # Memory
    "vision_weights_mb",
    "vlm_weights_mb",
    "action_weights_mb",
    "vlm_kv_cache_mb",
    "total_memory_mb",
    # Derived metrics
    "e2e_hz",
    "ttfa_ms",  # Time to first action = vision + vlm
]


@dataclass
class ComponentResult:
    """Result of evaluating a single model component."""
    time_ms: float
    boundness: str
    op_intensity: float
    weights_mb: float
    kv_cache_mb: float = 0.0


class VLAPerfEngine:
    """Performance evaluation engine for VLA models."""

    def __init__(self, batch_size: int = 1, bits: str = "bf16"):
        self.batch_size = batch_size
        self.bits = bits

    def _get_model(self, model_name: str):
        """Get model config from GenZ registry."""
        return get_configs(model_name)

    def _run_prefill(
        self, model_name: str, system: str, input_tokens: int,
        num_devices: int = 1, batch_multiplier: int = 1,
    ) -> ComponentResult:
        """Run prefill modeling for a component (vision encoder or VLM backbone)."""
        bits = get_best_precision_for_system(system, self.bits)
        model = self._get_model(model_name)
        effective_bs = self.batch_size * batch_multiplier

        parallelisms = get_parallelism(num_devices, system=system)
        best_result = None

        for tp, pp in parallelisms:
            try:
                output = prefill_moddeling(
                    model=model,
                    batch_size=effective_bs,
                    input_tokens=input_tokens,
                    system_name=system,
                    bits=bits,
                    tensor_parallel=tp,
                    pipeline_parallel=pp,
                    debug=False,
                )
                boundness, op_intensity = evaluate_boundness(output["model_df"])
                unit = Unit()
                st = output.get("summary_table")
                w_mb = st[f'Total Weights ({unit.unit_mem})'].values[0] if st is not None else 0
                kv_mb = st[f'KV Cache ({unit.unit_mem})'].values[0] if st is not None else 0

                if best_result is None or output["Latency"] < best_result.time_ms:
                    best_result = ComponentResult(
                        time_ms=output["Latency"],
                        boundness=boundness,
                        op_intensity=op_intensity,
                        weights_mb=w_mb,
                        kv_cache_mb=kv_mb,
                    )
            except Exception as e:
                logger.debug(f"Prefill failed for {model_name} on {system} tp={tp} pp={pp}: {e}")

        if best_result is None:
            raise RuntimeError(f"All parallelism configs failed for {model_name} on {system}")
        return best_result

    def _run_decode(
        self, model_name: str, system: str, input_tokens: int,
        output_tokens: int, num_devices: int = 1,
    ) -> ComponentResult:
        """Run decode modeling (for AR action heads)."""
        bits = get_best_precision_for_system(system, self.bits)
        model = self._get_model(model_name)

        parallelisms = get_parallelism(num_devices, system=system)
        best_result = None

        for tp, pp in parallelisms:
            try:
                output = decode_moddeling(
                    model=model,
                    batch_size=self.batch_size,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    Bb=1,
                    system_name=system,
                    bits=bits,
                    tensor_parallel=tp,
                    pipeline_parallel=pp,
                    debug=False,
                )
                boundness, op_intensity = evaluate_boundness(output["model_df"])
                unit = Unit()
                st = output.get("summary_table")
                w_mb = st[f'Total Weights ({unit.unit_mem})'].values[0] if st is not None else 0
                kv_mb = st[f'KV Cache ({unit.unit_mem})'].values[0] if st is not None else 0

                if best_result is None or output["Latency"] < best_result.time_ms:
                    best_result = ComponentResult(
                        time_ms=output["Latency"],
                        boundness=boundness,
                        op_intensity=op_intensity,
                        weights_mb=w_mb,
                        kv_cache_mb=kv_mb,
                    )
            except Exception as e:
                logger.debug(f"Decode failed for {model_name} on {system} tp={tp} pp={pp}: {e}")

        if best_result is None:
            raise RuntimeError(f"All parallelism configs failed for {model_name} on {system}")
        return best_result

    def _run_parallel_decode(
        self, model_name: str, system: str, input_tokens: int,
        output_tokens_parallel: int, self_attention: bool = True,
        num_devices: int = 1,
    ) -> ComponentResult:
        """Run parallel decode modeling (for FM/Diff action heads)."""
        bits = get_best_precision_for_system(system, self.bits)
        model = self._get_model(model_name)

        parallelisms = get_parallelism(num_devices, system=system)
        best_result = None

        for tp, pp in parallelisms:
            try:
                output = parallel_decode_modeling(
                    model=model,
                    batch_size=self.batch_size,
                    input_tokens=input_tokens,
                    output_tokens_parallel=output_tokens_parallel,
                    self_attention=self_attention,
                    system_name=system,
                    bits=bits,
                    tensor_parallel=tp,
                    pipeline_parallel=pp,
                    debug=False,
                )
                boundness, op_intensity = evaluate_boundness(output["model_df"])
                unit = Unit()
                st = output.get("summary_table")
                w_mb = st[f'Total Weights ({unit.unit_mem})'].values[0] if st is not None else 0
                kv_mb = st[f'KV Cache ({unit.unit_mem})'].values[0] if st is not None else 0

                if best_result is None or output["Latency"] < best_result.time_ms:
                    best_result = ComponentResult(
                        time_ms=output["Latency"],
                        boundness=boundness,
                        op_intensity=op_intensity,
                        weights_mb=w_mb,
                        kv_cache_mb=kv_mb,
                    )
            except Exception as e:
                logger.debug(f"Parallel decode failed for {model_name} on {system} tp={tp} pp={pp}: {e}")

        if best_result is None:
            raise RuntimeError(f"All parallelism configs failed for {model_name} on {system}")
        return best_result

    # ================================================================
    # Component-level evaluation
    # ================================================================

    def evaluate_vision(
        self, config: VLAConfig, system: str, num_devices: int = 1,
    ) -> ComponentResult:
        """Evaluate vision encoder latency."""
        v = config.vision
        return self._run_prefill(
            model_name=v["model_name"],
            system=system,
            input_tokens=v["tokens_per_frame"],
            num_devices=num_devices,
            batch_multiplier=config.num_frames,
        )

    def evaluate_vlm(
        self, config: VLAConfig, system: str, num_devices: int = 1,
    ) -> ComponentResult:
        """Evaluate VLM backbone prefill latency."""
        l = config.language
        return self._run_prefill(
            model_name=l["model_name"],
            system=system,
            input_tokens=config.vlm_sequence_length,
            num_devices=num_devices,
        )

    def evaluate_action(
        self, config: VLAConfig, system: str, num_devices: int = 1,
    ) -> ComponentResult:
        """Evaluate action head latency based on action type."""
        a = config.action
        action_type = a["type"]
        chunk_size = config.effective_chunk_size
        denoising_steps = config.effective_denoising_steps

        if action_type == "cascade_denoise":
            # Cascade: VLM produces features → separate DiT cross-attends → N steps
            # Each step: DiT processes action tokens with cross-attention to VLM output
            # VLM KV cache is the "input_tokens" context for the DiT
            single_step = self._run_parallel_decode(
                model_name=a["model_name"],
                system=system,
                input_tokens=config.vlm_sequence_length,
                output_tokens_parallel=chunk_size,
                self_attention=True,
                num_devices=num_devices,
            )
            return ComponentResult(
                time_ms=single_step.time_ms * denoising_steps,
                boundness=single_step.boundness,
                op_intensity=single_step.op_intensity,
                weights_mb=single_step.weights_mb,
                kv_cache_mb=single_step.kv_cache_mb,
            )

        elif action_type == "shared_attn_denoise":
            # SharedAttn (pi0-style): action tokens enter VLM's self-attention
            # KV cache from VLM observation tokens is reused across denoising steps
            # Per step: only action tokens are recomputed (parallel decode on VLM)
            # Use VLM backbone model (not separate DiT) for the shared attention pass
            l = config.language
            single_step = self._run_parallel_decode(
                model_name=l["model_name"],
                system=system,
                input_tokens=config.vlm_sequence_length,
                output_tokens_parallel=chunk_size,
                self_attention=True,
                num_devices=num_devices,
            )
            return ComponentResult(
                time_ms=single_step.time_ms * denoising_steps,
                boundness=single_step.boundness,
                op_intensity=single_step.op_intensity,
                weights_mb=single_step.weights_mb,
                kv_cache_mb=single_step.kv_cache_mb,
            )

        elif action_type == "cross_attn_denoise":
            # CrossAttn (SmolVLA-style): separate lightweight DiT with cross-attn
            # Similar to cascade but DiT is independent; VLM features cached
            single_step = self._run_parallel_decode(
                model_name=a["model_name"],
                system=system,
                input_tokens=config.vlm_sequence_length,
                output_tokens_parallel=chunk_size,
                self_attention=True,
                num_devices=num_devices,
            )
            return ComponentResult(
                time_ms=single_step.time_ms * denoising_steps,
                boundness=single_step.boundness,
                op_intensity=single_step.op_intensity,
                weights_mb=single_step.weights_mb,
                kv_cache_mb=single_step.kv_cache_mb,
            )

        elif action_type == "autoregressive":
            # AR: VLM directly generates discrete action tokens one at a time
            l = config.language
            single_token = self._run_decode(
                model_name=l["model_name"],
                system=system,
                input_tokens=config.vlm_sequence_length,
                output_tokens=1,
                num_devices=num_devices,
            )
            return ComponentResult(
                time_ms=single_token.time_ms * chunk_size,
                boundness=single_token.boundness,
                op_intensity=single_token.op_intensity,
                weights_mb=single_token.weights_mb,
                kv_cache_mb=single_token.kv_cache_mb,
            )

        elif action_type == "regression":
            # Regression: VLM hidden → MLP → continuous actions (single pass)
            return self._run_prefill(
                model_name=a["model_name"],
                system=system,
                input_tokens=1,
                num_devices=num_devices,
            )

        else:
            raise ValueError(f"Unknown action type: {action_type}")

    # ================================================================
    # End-to-end evaluation
    # ================================================================

    def evaluate_e2e(
        self, config: VLAConfig, system: str, num_devices: int = 1,
    ) -> dict:
        """
        Evaluate end-to-end VLA inference latency with per-component breakdown.

        Returns a dict with all BENCH_COLUMNS fields.
        """
        # Check memory fit
        fits, reason = self.check_memory_fit(config, system, num_devices)
        if not fits:
            logger.warning(f"Config {config.config_id} ({config.display_name}) "
                           f"does not fit on {system}: {reason}")
            return None

        # Evaluate each component
        try:
            vision_result = self.evaluate_vision(config, system, num_devices)
        except Exception as e:
            logger.error(f"Vision eval failed for config {config.config_id}: {e}")
            return None

        try:
            vlm_result = self.evaluate_vlm(config, system, num_devices)
        except Exception as e:
            logger.error(f"VLM eval failed for config {config.config_id}: {e}")
            return None

        try:
            action_result = self.evaluate_action(config, system, num_devices)
        except Exception as e:
            logger.error(f"Action eval failed for config {config.config_id}: {e}")
            return None

        # Compute derived metrics
        e2e_ms = vision_result.time_ms + vlm_result.time_ms + action_result.time_ms
        ttfa_ms = vision_result.time_ms + vlm_result.time_ms
        e2e_hz = 1000.0 / e2e_ms if e2e_ms > 0 else 0.0
        total_mem = (vision_result.weights_mb + vlm_result.weights_mb +
                     action_result.weights_mb + vlm_result.kv_cache_mb +
                     action_result.kv_cache_mb)

        return {
            "config_id": config.config_id,
            "config_name": config.display_name,
            "group": config.group,
            "phase": config.phase,
            "vision_key": config.vision_key,
            "language_key": config.language_key,
            "action_key": config.action_key,
            "action_type": config.action_type,
            "hardware": system,
            "num_devices": num_devices,
            "bits": self.bits,
            "batch_size": self.batch_size,
            "chunk_size": config.effective_chunk_size,
            "denoising_steps": config.effective_denoising_steps,
            "num_frames": config.num_frames,
            "vision_tokens": config.vision_tokens,
            "vlm_seq_len": config.vlm_sequence_length,
            "vision_time_ms": vision_result.time_ms,
            "vlm_time_ms": vlm_result.time_ms,
            "action_time_ms": action_result.time_ms,
            "e2e_time_ms": e2e_ms,
            "vision_boundness": vision_result.boundness,
            "vlm_boundness": vlm_result.boundness,
            "action_boundness": action_result.boundness,
            "vision_op_intensity": vision_result.op_intensity,
            "vlm_op_intensity": vlm_result.op_intensity,
            "action_op_intensity": action_result.op_intensity,
            "vision_weights_mb": vision_result.weights_mb,
            "vlm_weights_mb": vlm_result.weights_mb,
            "action_weights_mb": action_result.weights_mb,
            "vlm_kv_cache_mb": vlm_result.kv_cache_mb,
            "total_memory_mb": total_mem,
            "e2e_hz": e2e_hz,
            "ttfa_ms": ttfa_ms,
        }

    def check_memory_fit(
        self, config: VLAConfig, system: str, num_devices: int = 1,
    ) -> tuple:
        """
        Check if a VLA config fits in the system memory.

        Returns (fits: bool, reason: str)
        """
        if system not in system_configs:
            return True, "Unknown system, assuming fits"

        sys_mem_gb = system_configs[system]["Memory_size"] * num_devices
        bits = get_best_precision_for_system(system, self.bits)
        bytes_per_param = {"bf16": 2, "fp16": 2, "fp32": 4, "int8": 1, "int4": 0.5}.get(bits, 2)

        total_params = 0
        # Vision encoder
        v_model = self._get_model(config.vision["model_name"])
        total_params += calculate_transformer_params(v_model)

        # Language backbone
        l_model = self._get_model(config.language["model_name"])
        total_params += calculate_transformer_params(l_model)

        # Action head (if not AR, which reuses LLM)
        if config.action["model_name"] is not None:
            a_model = self._get_model(config.action["model_name"])
            total_params += calculate_transformer_params(a_model)

        weights_gb = (total_params * bytes_per_param) / (1024**3)
        # KV cache estimate (rough: 2 * layers * 2 * heads * head_dim * seq_len * bytes)
        kv_gb = 0.5  # Conservative estimate for batch_size=1

        total_gb = weights_gb + kv_gb
        fits = total_gb <= sys_mem_gb

        if not fits:
            return False, f"Need {total_gb:.1f}GB, have {sys_mem_gb}GB"
        return True, f"OK ({total_gb:.1f}GB / {sys_mem_gb}GB)"

    # ================================================================
    # Batch evaluation
    # ================================================================

    def evaluate_all(
        self,
        configs: list,
        systems: list,
        num_devices: int = 1,
        output_dir: str = "results",
    ) -> pd.DataFrame:
        """
        Evaluate all VLA configs on all systems.

        Args:
            configs: List of VLAConfig objects
            systems: List of system names
            num_devices: Number of devices per system
            output_dir: Directory to save results

        Returns:
            DataFrame with all results
        """
        results = []
        total = len(configs) * len(systems)
        done = 0

        for system in systems:
            for config in configs:
                done += 1
                logger.info(f"[{done}/{total}] Config #{config.config_id} "
                           f"({config.display_name}) on {system}")
                try:
                    result = self.evaluate_e2e(config, system, num_devices)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"  FAILED: {e}")

        df = pd.DataFrame(results)

        if not df.empty:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            csv_path = output_path / "vla_benchmark_results.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}")

        return df
