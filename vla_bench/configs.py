"""
VLA Configuration Registry

Defines all 64 VLA configurations across 3 experiment phases (P0/P1/P2).

Model components:
  Vision (V): SigLIP2-B (86M), SigLIP2-L (307M), SigLIP2-So (400M)
  Language (L): Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B
  Action (A): FM (Flow Matching), Diff (Diffusion), AR (Autoregressive), MLP

Action head sizes:
  FM/Diff: S (~50M), M (~200M), L (~450M)
  AR: reuses LLM backbone (no separate action model)
  MLP: S (~10M), M (~30M), L (~80M)
"""

from dataclasses import dataclass
from typing import Optional


# ============================================================
# Vision encoder constants
# ============================================================

VISION_CONFIGS = {
    "V-S": {
        "model_name": "siglip2-base-patch16-224-vision",
        "display_name": "SigLIP2-B",
        "params": "86M",
        "tokens_per_frame": 256,
        "resolution": 224,
    },
    "V-M": {
        "model_name": "siglip2-large-patch16-384-vision",
        "display_name": "SigLIP2-L",
        "params": "307M",
        "tokens_per_frame": 576,
        "resolution": 384,
    },
    "V-L": {
        "model_name": "siglip2-so400m-patch14-384-vision",
        "display_name": "SigLIP2-So",
        "params": "400M",
        "tokens_per_frame": 256,
        "resolution": 384,
    },
}

# ============================================================
# Language backbone constants
# ============================================================

LANGUAGE_CONFIGS = {
    "L-S": {
        "model_name": "vla/qwen2.5-0.5b",
        "display_name": "Qwen2.5-0.5B",
        "params": "0.5B",
    },
    "L-M": {
        "model_name": "vla/qwen2.5-1.5b",
        "display_name": "Qwen2.5-1.5B",
        "params": "1.5B",
    },
    "L-L": {
        "model_name": "vla/qwen2.5-3b",
        "display_name": "Qwen2.5-3B",
        "params": "3B",
    },
}

# ============================================================
# Action head constants
# ============================================================

ACTION_CONFIGS = {
    # Flow Matching: DiT with cross-attn to VLM KV cache, N denoising steps
    "FM-S": {
        "model_name": "fm-action-expert-s",
        "display_name": "FM-S",
        "type": "flow_matching",
        "params": "50M",
        "default_denoising_steps": 10,
        "default_chunk_size": 10,
    },
    "FM-M": {
        "model_name": "fm-action-expert-m",
        "display_name": "FM-M",
        "type": "flow_matching",
        "params": "200M",
        "default_denoising_steps": 10,
        "default_chunk_size": 10,
    },
    "FM-L": {
        "model_name": "fm-action-expert-l",
        "display_name": "FM-L",
        "type": "flow_matching",
        "params": "450M",
        "default_denoising_steps": 10,
        "default_chunk_size": 10,
    },
    # Diffusion: DiT with cross-attn, DDPM schedule, N denoising steps
    "Diff-S": {
        "model_name": "diff-action-expert-s",
        "display_name": "Diff-S",
        "type": "diffusion",
        "params": "50M",
        "default_denoising_steps": 10,
        "default_chunk_size": 10,
    },
    "Diff-M": {
        "model_name": "diff-action-expert-m",
        "display_name": "Diff-M",
        "type": "diffusion",
        "params": "200M",
        "default_denoising_steps": 10,
        "default_chunk_size": 10,
    },
    "Diff-L": {
        "model_name": "diff-action-expert-l",
        "display_name": "Diff-L",
        "type": "diffusion",
        "params": "450M",
        "default_denoising_steps": 10,
        "default_chunk_size": 10,
    },
    # Autoregressive: reuses LLM backbone, sequential token generation
    "AR": {
        "model_name": None,  # reuses LLM backbone
        "display_name": "AR",
        "type": "autoregressive",
        "params": "0",  # shared with LLM
        "default_denoising_steps": 1,
        "default_chunk_size": 10,
    },
    # MLP: simple projection head
    "MLP-S": {
        "model_name": "mlp-action-head-s",
        "display_name": "MLP-S",
        "type": "mlp",
        "params": "10M",
        "default_denoising_steps": 1,
        "default_chunk_size": 10,
    },
    "MLP-M": {
        "model_name": "mlp-action-head-m",
        "display_name": "MLP-M",
        "type": "mlp",
        "params": "30M",
        "default_denoising_steps": 1,
        "default_chunk_size": 10,
    },
    "MLP-L": {
        "model_name": "mlp-action-head-l",
        "display_name": "MLP-L",
        "type": "mlp",
        "params": "80M",
        "default_denoising_steps": 1,
        "default_chunk_size": 10,
    },
}


# ============================================================
# VLA configuration dataclass
# ============================================================

@dataclass
class VLAConfig:
    """A complete VLA model configuration."""
    config_id: int                    # Unique ID (1-64)
    group: str                        # Experiment group (A-M)
    phase: str                        # Experiment phase (P0/P1/P2)

    # Component keys (index into *_CONFIGS dicts)
    vision_key: str                   # "V-S", "V-M", "V-L"
    language_key: str                 # "L-S", "L-M", "L-L"
    action_key: str                   # "FM-S", ..., "AR", "MLP-S", ...

    # Override defaults for sweep experiments
    chunk_size: Optional[int] = None
    denoising_steps: Optional[int] = None

    # Inference parameters
    num_frames: int = 1               # Number of camera frames
    language_tokens: int = 32         # Language instruction tokens
    action_dof: int = 7               # Degrees of freedom

    @property
    def vision(self):
        return VISION_CONFIGS[self.vision_key]

    @property
    def language(self):
        return LANGUAGE_CONFIGS[self.language_key]

    @property
    def action(self):
        return ACTION_CONFIGS[self.action_key]

    @property
    def action_type(self):
        return self.action["type"]

    @property
    def effective_chunk_size(self):
        return self.chunk_size or self.action["default_chunk_size"]

    @property
    def effective_denoising_steps(self):
        return self.denoising_steps or self.action["default_denoising_steps"]

    @property
    def vision_tokens(self):
        """Total vision tokens = tokens_per_frame * num_frames."""
        return self.vision["tokens_per_frame"] * self.num_frames

    @property
    def vlm_sequence_length(self):
        """Total input tokens to VLM = vision_tokens + language_tokens."""
        return self.vision_tokens + self.language_tokens

    @property
    def display_name(self):
        v = self.vision["display_name"]
        l = self.language["display_name"]
        a = self.action["display_name"]
        suffix = ""
        if self.chunk_size is not None:
            suffix += f",c={self.chunk_size}"
        if self.denoising_steps is not None and self.action_type in ("flow_matching", "diffusion"):
            suffix += f",s={self.denoising_steps}"
        return f"{v}+{l}+{a}{suffix}"

    @property
    def vlm_id(self):
        """VLM identifier (vision + language combination)."""
        return f"{self.vision_key}_{self.language_key}"


# ============================================================
# VLM matrix (3x3 = 9 VLMs)
# ============================================================

# VLM numbering: vision rows x language columns
#            L-S(0.5B)  L-M(1.5B)  L-L(3B)
# V-S(B)      VLM-1      VLM-2     VLM-3
# V-M(L)      VLM-4      VLM-5*    VLM-6
# V-L(So)     VLM-7      VLM-8     VLM-9

VLM_MATRIX = {
    1: ("V-S", "L-S"),  # SigLIP2-B + Qwen2.5-0.5B  ~0.6B
    2: ("V-S", "L-M"),  # SigLIP2-B + Qwen2.5-1.5B  ~1.6B
    3: ("V-S", "L-L"),  # SigLIP2-B + Qwen2.5-3B    ~3.1B
    4: ("V-M", "L-S"),  # SigLIP2-L + Qwen2.5-0.5B  ~0.8B
    5: ("V-M", "L-M"),  # SigLIP2-L + Qwen2.5-1.5B  ~1.8B (baseline)
    6: ("V-M", "L-L"),  # SigLIP2-L + Qwen2.5-3B    ~3.3B
    7: ("V-L", "L-S"),  # SigLIP2-So + Qwen2.5-0.5B ~0.9B
    8: ("V-L", "L-M"),  # SigLIP2-So + Qwen2.5-1.5B ~1.9B
    9: ("V-L", "L-L"),  # SigLIP2-So + Qwen2.5-3B   ~3.4B
}


def _vlm(vlm_num):
    """Helper: return (vision_key, language_key) for a VLM number."""
    return VLM_MATRIX[vlm_num]


# ============================================================
# P0: 40 VLAs -> Q1-Q4
# ============================================================

def _build_p0_configs():
    """Build P0 experiment configs (40 VLAs)."""
    configs = []

    # Group A: 3x3 V×L grid, fixed FM-M action head (9 configs)
    for i, vlm_num in enumerate(range(1, 10), start=1):
        v, l = _vlm(vlm_num)
        configs.append(VLAConfig(
            config_id=i, group="A", phase="P0",
            vision_key=v, language_key=l, action_key="FM-M",
        ))

    # Group B: V-Scaling × Diff/AR/MLP (fix L=1.5B) (9 configs)
    b_id = 10
    for v_key in ["V-S", "V-M", "V-L"]:
        for action_key in ["Diff-M", "AR", "MLP-M"]:
            configs.append(VLAConfig(
                config_id=b_id, group="B", phase="P0",
                vision_key=v_key, language_key="L-M", action_key=action_key,
            ))
            b_id += 1

    # Group C: L-Scaling × Diff/AR/MLP (fix V=L, SigLIP2-L) (6 configs)
    c_id = 19
    for l_key in ["L-S", "L-L"]:
        for action_key in ["Diff-M", "AR", "MLP-M"]:
            configs.append(VLAConfig(
                config_id=c_id, group="C", phase="P0",
                vision_key="V-M", language_key=l_key, action_key=action_key,
            ))
            c_id += 1

    # Group D: A-Scaling on VLM-5 (FM/Diff/MLP × S/M/L) (6 configs)
    d_id = 25
    for action_key in ["FM-S", "FM-L", "Diff-S", "Diff-L", "MLP-S", "MLP-L"]:
        configs.append(VLAConfig(
            config_id=d_id, group="D", phase="P0",
            vision_key="V-M", language_key="L-M", action_key=action_key,
        ))
        d_id += 1

    # Group E: Corner VLMs (1,9) × Diff/AR/MLP (6 configs)
    e_id = 31
    for vlm_num in [1, 9]:
        v, l = _vlm(vlm_num)
        for action_key in ["Diff-M", "AR", "MLP-M"]:
            configs.append(VLAConfig(
                config_id=e_id, group="E", phase="P0",
                vision_key=v, language_key=l, action_key=action_key,
            ))
            e_id += 1

    # Group F: FM A-Scaling across VLMs 1,9 (4 configs)
    f_id = 37
    for vlm_num in [1, 9]:
        v, l = _vlm(vlm_num)
        for action_key in ["FM-S", "FM-L"]:
            configs.append(VLAConfig(
                config_id=f_id, group="F", phase="P0",
                vision_key=v, language_key=l, action_key=action_key,
            ))
            f_id += 1

    assert len(configs) == 40, f"Expected 40 P0 configs, got {len(configs)}"
    return configs


# ============================================================
# P1: 12 VLAs -> Q5
# ============================================================

def _build_p1_configs():
    """Build P1 experiment configs (12 VLAs)."""
    configs = []

    # Group G: Chunk size sweep (VLM-5 + FM-M) (4 configs)
    g_id = 41
    for chunk in [1, 5, 25, 50]:
        configs.append(VLAConfig(
            config_id=g_id, group="G", phase="P1",
            vision_key="V-M", language_key="L-M", action_key="FM-M",
            chunk_size=chunk,
        ))
        g_id += 1

    # Group H: Denoising steps sweep (VLM-5 + FM-M) (3 configs)
    h_id = 45
    for steps in [5, 25, 50]:
        configs.append(VLAConfig(
            config_id=h_id, group="H", phase="P1",
            vision_key="V-M", language_key="L-M", action_key="FM-M",
            denoising_steps=steps,
        ))
        h_id += 1

    # Group I: AR chunk sweep (VLM-5 + AR) (3 configs)
    i_id = 48
    for chunk in [1, 5, 25]:
        configs.append(VLAConfig(
            config_id=i_id, group="I", phase="P1",
            vision_key="V-M", language_key="L-M", action_key="AR",
            chunk_size=chunk,
        ))
        i_id += 1

    # Group J: VLM-1/9 × MLP-S (2 configs)
    j_id = 51
    for vlm_num in [1, 9]:
        v, l = _vlm(vlm_num)
        configs.append(VLAConfig(
            config_id=j_id, group="J", phase="P1",
            vision_key=v, language_key=l, action_key="MLP-S",
        ))
        j_id += 1

    assert len(configs) == 12, f"Expected 12 P1 configs, got {len(configs)}"
    return configs


# ============================================================
# P2: 12 VLAs -> Q5-Q7 supplementary
# ============================================================

def _build_p2_configs():
    """Build P2 experiment configs (12 VLAs)."""
    configs = []

    # Group K: Chunk generalization VLM-1/9 × chunk={1,50} (4 configs)
    k_id = 53
    for vlm_num in [1, 9]:
        v, l = _vlm(vlm_num)
        for chunk in [1, 50]:
            configs.append(VLAConfig(
                config_id=k_id, group="K", phase="P2",
                vision_key=v, language_key=l, action_key="FM-M",
                chunk_size=chunk,
            ))
            k_id += 1

    # Group L: Steps generalization VLM-1/9 × steps={5,50} (4 configs)
    l_id = 57
    for vlm_num in [1, 9]:
        v, l_ = _vlm(vlm_num)
        for steps in [5, 50]:
            configs.append(VLAConfig(
                config_id=l_id, group="L", phase="P2",
                vision_key=v, language_key=l_, action_key="FM-M",
                denoising_steps=steps,
            ))
            l_id += 1

    # Group M: Diff scaling generalization VLM-1/9 × Diff-S/L (4 configs)
    m_id = 61
    for vlm_num in [1, 9]:
        v, l = _vlm(vlm_num)
        for action_key in ["Diff-S", "Diff-L"]:
            configs.append(VLAConfig(
                config_id=m_id, group="M", phase="P2",
                vision_key=v, language_key=l, action_key=action_key,
            ))
            m_id += 1

    assert len(configs) == 12, f"Expected 12 P2 configs, got {len(configs)}"
    return configs


# ============================================================
# Public API
# ============================================================

_ALL_CONFIGS = None


def get_all_configs():
    """Return all 64 VLA configurations."""
    global _ALL_CONFIGS
    if _ALL_CONFIGS is None:
        _ALL_CONFIGS = _build_p0_configs() + _build_p1_configs() + _build_p2_configs()
        assert len(_ALL_CONFIGS) == 64, f"Expected 64 configs, got {len(_ALL_CONFIGS)}"
    return _ALL_CONFIGS


def get_config_by_id(config_id: int) -> VLAConfig:
    """Get a specific VLA config by its ID (1-64)."""
    for cfg in get_all_configs():
        if cfg.config_id == config_id:
            return cfg
    raise ValueError(f"Config ID {config_id} not found (valid: 1-64)")


def get_configs_by_group(group: str) -> list:
    """Get all configs in a specific experiment group (A-M)."""
    return [c for c in get_all_configs() if c.group == group]


def get_configs_by_phase(phase: str) -> list:
    """Get all configs in a specific phase (P0/P1/P2)."""
    return [c for c in get_all_configs() if c.phase == phase]


# ============================================================
# Hardware presets
# ============================================================

HARDWARE_PRESETS = {
    "edge": ["Jetson_AGX_Orin_64GB"],
    "server": ["A800_80GB"],
    "full": ["Jetson_AGX_Orin_64GB", "A800_80GB", "H100", "RTX_4090"],
    "datacenter": ["A100_80GB", "A800_80GB", "H100", "B100"],
    "jetson": [
        "Jetson_Orin_Nano_8GB", "Jetson_Orin_NX_16GB",
        "Jetson_AGX_Orin_32GB", "Jetson_AGX_Orin_64GB", "Jetson_AGX_Thor",
    ],
}

# Default test platforms from the paper
DEFAULT_SYSTEMS = ["Jetson_AGX_Orin_64GB", "A800_80GB"]

# Precision configs
PRECISION_CONFIGS = {
    "default": "bf16",
    "quantized": ["bf16", "int8", "int4"],
}
