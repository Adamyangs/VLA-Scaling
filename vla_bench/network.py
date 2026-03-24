"""
Network Latency Analysis for VLA Deployment Scenarios

Extends VLA-Perf's network modeling for split inference scenarios:
  - On-device: no network overhead
  - Edge server: device sends images, receives actions (WiFi/Ethernet)
  - Cloud server: higher latency WAN links
  - Split inference: vision on device, VLM+action on server
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

_VLA_PERF_DIR = Path(__file__).resolve().parent.parent.parent / "vla-perf" / "vla-perf"
if str(_VLA_PERF_DIR) not in sys.path:
    sys.path.insert(0, str(_VLA_PERF_DIR))

from network_latency import (
    NetworkConfig,
    ImageConfig,
    ActionConfig,
    estimate_image_latency,
    estimate_action_latency,
    WIFI_6_CONFIG,
    WIFI_7_CONFIG,
    ETHERNET_1G_CONFIG,
    ETHERNET_10G_CONFIG,
)

from .configs import VLAConfig


@dataclass
class DeploymentScenario:
    """A deployment scenario for VLA inference."""
    name: str
    description: str
    # Where each component runs
    vision_device: str     # "edge" or "server"
    vlm_device: str        # "edge" or "server"
    action_device: str     # "edge" or "server"
    # Network between edge and server
    network: Optional[NetworkConfig] = None

    @property
    def requires_network(self):
        """Network is needed if any component runs off the edge device."""
        return any(d != "edge" for d in
                   [self.vision_device, self.vlm_device, self.action_device])


# Pre-defined deployment scenarios
SCENARIOS = {
    "on_device": DeploymentScenario(
        name="on_device",
        description="All components on edge device",
        vision_device="edge", vlm_device="edge", action_device="edge",
    ),
    "full_offload_wifi6": DeploymentScenario(
        name="full_offload_wifi6",
        description="Send images to server, receive actions via WiFi 6",
        vision_device="server", vlm_device="server", action_device="server",
        network=WIFI_6_CONFIG,
    ),
    "full_offload_wifi7": DeploymentScenario(
        name="full_offload_wifi7",
        description="Send images to server, receive actions via WiFi 7",
        vision_device="server", vlm_device="server", action_device="server",
        network=WIFI_7_CONFIG,
    ),
    "full_offload_eth1g": DeploymentScenario(
        name="full_offload_eth1g",
        description="Send images to server via 1G Ethernet",
        vision_device="server", vlm_device="server", action_device="server",
        network=ETHERNET_1G_CONFIG,
    ),
    "full_offload_eth10g": DeploymentScenario(
        name="full_offload_eth10g",
        description="Send images to server via 10G Ethernet",
        vision_device="server", vlm_device="server", action_device="server",
        network=ETHERNET_10G_CONFIG,
    ),
    "split_vision_wifi6": DeploymentScenario(
        name="split_vision_wifi6",
        description="Vision on edge, VLM+Action on server via WiFi 6",
        vision_device="edge", vlm_device="server", action_device="server",
        network=WIFI_6_CONFIG,
    ),
    "split_vision_wifi7": DeploymentScenario(
        name="split_vision_wifi7",
        description="Vision on edge, VLM+Action on server via WiFi 7",
        vision_device="edge", vlm_device="server", action_device="server",
        network=WIFI_7_CONFIG,
    ),
    "split_vision_eth10g": DeploymentScenario(
        name="split_vision_eth10g",
        description="Vision on edge, VLM+Action on server via 10G Ethernet",
        vision_device="edge", vlm_device="server", action_device="server",
        network=ETHERNET_10G_CONFIG,
    ),
}


def estimate_image_transfer_ms(
    config: VLAConfig,
    network: NetworkConfig,
    compression_ratio: float = 0.1,
) -> float:
    """
    Estimate image upload latency from edge to server.

    Args:
        config: VLA configuration
        network: Network configuration
        compression_ratio: JPEG compression ratio (0.1 = 10x compression)

    Returns:
        Transfer time in milliseconds
    """
    resolution = config.vision["resolution"]
    img = ImageConfig(
        resolution=resolution,
        channels=3,
        bytes_per_pixel=1,
        compression_ratio=compression_ratio,
    )
    result = estimate_image_latency(network, img)
    # Multiply by num_frames
    return result["total_latency_ms"] * config.num_frames


def estimate_action_transfer_ms(
    config: VLAConfig,
    network: NetworkConfig,
) -> float:
    """
    Estimate action download latency from server to edge.

    Args:
        config: VLA configuration
        network: Network configuration

    Returns:
        Transfer time in milliseconds
    """
    action = ActionConfig(
        num_dof=config.action_dof,
        action_chunk_size=config.effective_chunk_size,
        bytes_per_value=4,  # float32
    )
    result = estimate_action_latency(network, action)
    return result["total_latency_ms"]


def estimate_vision_features_transfer_ms(
    config: VLAConfig,
    network: NetworkConfig,
) -> float:
    """
    Estimate vision feature transfer latency for split inference.
    Vision features are the output tokens from the vision encoder.

    Args:
        config: VLA configuration
        network: Network configuration

    Returns:
        Transfer time in milliseconds
    """
    # Vision features: num_tokens × hidden_size × bytes_per_element
    # Approximate hidden size based on vision encoder
    hidden_sizes = {
        "V-S": 768,   # SigLIP2-B
        "V-M": 1024,  # SigLIP2-L
        "V-L": 1152,  # SigLIP2-So
    }
    hidden = hidden_sizes.get(config.vision_key, 1024)
    num_tokens = config.vision_tokens
    bytes_per_element = 2  # bf16

    feature_bytes = num_tokens * hidden * bytes_per_element
    feature_bits = feature_bytes * 8

    # Upload: edge → server
    bw_upload = network.bandwidth_mbps("upload") * 1e6  # bits per second
    transfer_s = feature_bits / bw_upload if bw_upload > 0 else float('inf')
    return network.base_latency_ms + transfer_s * 1000


def estimate_deployment_latency(
    config: VLAConfig,
    scenario: DeploymentScenario,
    edge_result: dict = None,
    server_result: dict = None,
    compression_ratio: float = 0.1,
) -> dict:
    """
    Estimate total latency for a deployment scenario including network overhead.

    For on-device: only edge_result is needed.
    For full offload: only server_result is needed.
    For split inference: both edge_result and server_result are needed.

    Args:
        config: VLA configuration
        scenario: Deployment scenario
        edge_result: Result dict from evaluate_e2e() on edge hardware
        server_result: Result dict from evaluate_e2e() on server hardware
        compression_ratio: Image compression ratio

    Returns:
        Dict with deployment latency breakdown
    """
    result = {
        "scenario": scenario.name,
        "description": scenario.description,
    }

    if not scenario.requires_network:
        # All on device - no network overhead
        if edge_result is None:
            raise ValueError("on-device scenario requires edge_result")
        result["network_upload_ms"] = 0
        result["network_download_ms"] = 0
        result["compute_ms"] = edge_result["e2e_time_ms"]
        result["total_latency_ms"] = edge_result["e2e_time_ms"]
        return result

    network = scenario.network

    if scenario.vision_device == "server":
        # Full offload: send image to server, receive action back
        if server_result is None:
            raise ValueError("full offload scenario requires server_result")
        upload_ms = estimate_image_transfer_ms(config, network, compression_ratio)
        download_ms = estimate_action_transfer_ms(config, network)
        compute_ms = server_result["e2e_time_ms"]
    else:
        # Split: vision on edge, VLM+action on server
        if edge_result is None or server_result is None:
            raise ValueError("split scenario requires both edge_result and server_result")
        upload_ms = estimate_vision_features_transfer_ms(config, network)
        download_ms = estimate_action_transfer_ms(config, network)
        # Vision runs on edge, VLM+Action run on server
        # Vision and upload can overlap; total = max(vision, upload) + VLM + Action
        edge_vision_ms = edge_result["vision_time_ms"]
        server_vlm_ms = server_result["vlm_time_ms"]
        server_action_ms = server_result["action_time_ms"]
        compute_ms = max(edge_vision_ms, upload_ms) + server_vlm_ms + server_action_ms
        # Upload is already accounted for in the overlap, set to 0 for breakdown
        upload_ms = 0

    result["network_upload_ms"] = upload_ms
    result["network_download_ms"] = download_ms
    result["compute_ms"] = compute_ms
    result["total_latency_ms"] = compute_ms + upload_ms + download_ms

    return result
