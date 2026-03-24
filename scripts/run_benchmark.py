#!/usr/bin/env python3
"""
VLA-Perf++ Main Benchmark Script

Runs all 64 VLA configurations on specified hardware platforms
and outputs per-component latency breakdown CSV.

Usage:
    # Run all 64 configs on default systems (Orin + A800)
    python scripts/run_benchmark.py

    # Run specific phase
    python scripts/run_benchmark.py --phase P0

    # Run specific config IDs
    python scripts/run_benchmark.py --configs 1 5 9

    # Run specific experiment group
    python scripts/run_benchmark.py --group A

    # Run on specific hardware
    python scripts/run_benchmark.py --systems Jetson_AGX_Orin_64GB A800_80GB H100

    # Run with quantization
    python scripts/run_benchmark.py --bits int8
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vla_bench.configs import (
    get_all_configs,
    get_config_by_id,
    get_configs_by_group,
    get_configs_by_phase,
    DEFAULT_SYSTEMS,
    HARDWARE_PRESETS,
)
from vla_bench.engine import VLAPerfEngine


def setup_logging(output_dir: str):
    """Configure logging to console and file."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_dir / "benchmark.log", mode="w"),
            logging.StreamHandler(),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="VLA-Perf++ Benchmark")

    # Config selection (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--configs", type=int, nargs="+",
                       help="Specific config IDs to run (1-64)")
    group.add_argument("--phase", type=str, choices=["P0", "P1", "P2"],
                       help="Run all configs in a phase")
    group.add_argument("--group", type=str,
                       help="Run all configs in a group (A-M)")
    group.add_argument("--all", action="store_true", default=True,
                       help="Run all 64 configs (default)")

    # Hardware selection
    parser.add_argument("--systems", type=str, nargs="+",
                        default=None,
                        help="Hardware systems to evaluate on")
    parser.add_argument("--preset", type=str, choices=list(HARDWARE_PRESETS.keys()),
                        default=None,
                        help="Use a hardware preset")

    # Inference parameters
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference (default: 1)")
    parser.add_argument("--bits", type=str, default="bf16",
                        choices=["bf16", "fp16", "int8", "int4"],
                        help="Precision (default: bf16)")
    parser.add_argument("--num-devices", type=int, default=1,
                        help="Number of devices (default: 1)")

    # Output
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results)")

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)

    # Select configs
    if args.configs:
        configs = [get_config_by_id(cid) for cid in args.configs]
        logger.info(f"Running {len(configs)} specific configs: {args.configs}")
    elif args.phase:
        configs = get_configs_by_phase(args.phase)
        logger.info(f"Running {len(configs)} configs from phase {args.phase}")
    elif args.group:
        configs = get_configs_by_group(args.group)
        logger.info(f"Running {len(configs)} configs from group {args.group}")
    else:
        configs = get_all_configs()
        logger.info(f"Running all {len(configs)} configs")

    # Select hardware
    if args.systems:
        systems = args.systems
    elif args.preset:
        systems = HARDWARE_PRESETS[args.preset]
    else:
        systems = DEFAULT_SYSTEMS

    logger.info(f"Hardware: {systems}")
    logger.info(f"Precision: {args.bits}, Batch size: {args.batch_size}, "
                f"Devices: {args.num_devices}")

    # Run benchmark
    engine = VLAPerfEngine(batch_size=args.batch_size, bits=args.bits)
    df = engine.evaluate_all(
        configs=configs,
        systems=systems,
        num_devices=args.num_devices,
        output_dir=args.output_dir,
    )

    if df.empty:
        logger.warning("No results produced!")
        return

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 80)

    for system in systems:
        sys_df = df[df["hardware"] == system]
        if sys_df.empty:
            continue

        logger.info(f"\n--- {system} ---")
        logger.info(f"{'ID':>3} {'Config':<40} {'Vision':>8} {'VLM':>8} "
                     f"{'Action':>8} {'E2E':>8} {'Hz':>6}")
        logger.info("-" * 85)

        for _, row in sys_df.iterrows():
            logger.info(
                f"{row['config_id']:>3} {row['config_name']:<40} "
                f"{row['vision_time_ms']:>7.2f}ms "
                f"{row['vlm_time_ms']:>7.2f}ms "
                f"{row['action_time_ms']:>7.2f}ms "
                f"{row['e2e_time_ms']:>7.2f}ms "
                f"{row['e2e_hz']:>5.1f}"
            )

    # Print fastest/slowest
    logger.info(f"\nFastest config: #{df.loc[df['e2e_time_ms'].idxmin(), 'config_id']} "
                f"({df['e2e_time_ms'].min():.2f}ms)")
    logger.info(f"Slowest config: #{df.loc[df['e2e_time_ms'].idxmax(), 'config_id']} "
                f"({df['e2e_time_ms'].max():.2f}ms)")

    logger.info(f"\nResults saved to {args.output_dir}/vla_benchmark_results.csv")


if __name__ == "__main__":
    main()
