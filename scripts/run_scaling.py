#!/usr/bin/env python3
"""
VLA Scaling Experiments

Runs targeted experiments for the 7 research questions:
  Q1: V/L/A component scaling efficiency
  Q2: Optimal V/L/A allocation under fixed parameter budget
  Q3: Bottleneck effects between components
  Q4: Action head architecture vs size
  Q5: Chunk size and denoising steps trade-off
  Q6: Cross-platform latency characteristics
  Q7: Accuracy-latency Pareto frontier (requires accuracy data)

Usage:
    python scripts/run_scaling.py --experiment q1
    python scripts/run_scaling.py --experiment all
"""

import sys
import argparse
import logging
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vla_bench.configs import (
    get_all_configs,
    get_configs_by_group,
    DEFAULT_SYSTEMS,
)
from vla_bench.engine import VLAPerfEngine

logger = logging.getLogger(__name__)


def run_experiment(engine, configs, systems, output_dir, name):
    """Run an experiment on given configs and save results."""
    df = engine.evaluate_all(configs, systems, output_dir=output_dir)
    if not df.empty:
        csv_path = Path(output_dir) / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {name} results to {csv_path} ({len(df)} rows)")
    return df


def q1_component_scaling(engine, systems, output_dir):
    """Q1: Which component benefits most from scaling?

    Groups B (V-Scaling), C (L-Scaling), D (A-Scaling)
    """
    logger.info("=" * 60)
    logger.info("Q1: Component Scaling Efficiency")
    logger.info("=" * 60)

    configs = get_configs_by_group("B") + get_configs_by_group("C") + get_configs_by_group("D")
    return run_experiment(engine, configs, systems, output_dir, "q1_component_scaling")


def q2_optimal_allocation(engine, systems, output_dir):
    """Q2: Optimal V/L/A allocation under fixed parameter budget.

    Group A: 3×3 V×L grid with fixed FM-M
    """
    logger.info("=" * 60)
    logger.info("Q2: Optimal V/L/A Allocation")
    logger.info("=" * 60)

    configs = get_configs_by_group("A")
    return run_experiment(engine, configs, systems, output_dir, "q2_vl_allocation")


def q3_bottleneck(engine, systems, output_dir):
    """Q3: Bottleneck effects between components.

    Use Group A corners (configs 1,3,7,9) + middle (5)
    """
    logger.info("=" * 60)
    logger.info("Q3: Bottleneck Analysis")
    logger.info("=" * 60)

    configs = [c for c in get_configs_by_group("A") if c.config_id in [1, 3, 5, 7, 9]]
    return run_experiment(engine, configs, systems, output_dir, "q3_bottleneck")


def q4_action_architecture(engine, systems, output_dir):
    """Q4: Action head architecture comparison.

    Groups D (A-Scaling on VLM-5), E (corner VLMs), F (FM scaling generalization)
    """
    logger.info("=" * 60)
    logger.info("Q4: Action Architecture Comparison")
    logger.info("=" * 60)

    configs = (get_configs_by_group("D") + get_configs_by_group("E") +
               get_configs_by_group("F"))
    return run_experiment(engine, configs, systems, output_dir, "q4_action_arch")


def q5_chunk_steps(engine, systems, output_dir):
    """Q5: Chunk size and denoising steps trade-off.

    Groups G (chunk sweep), H (steps sweep), I (AR chunk sweep)
    """
    logger.info("=" * 60)
    logger.info("Q5: Chunk Size & Denoising Steps")
    logger.info("=" * 60)

    configs = (get_configs_by_group("G") + get_configs_by_group("H") +
               get_configs_by_group("I"))
    return run_experiment(engine, configs, systems, output_dir, "q5_chunk_steps")


def q6_cross_platform(engine, systems, output_dir):
    """Q6: Cross-platform latency characteristics.

    All 64 configs on both edge and server platforms.
    """
    logger.info("=" * 60)
    logger.info("Q6: Cross-Platform Analysis")
    logger.info("=" * 60)

    configs = get_all_configs()
    return run_experiment(engine, configs, systems, output_dir, "q6_cross_platform")


def q5_supplementary(engine, systems, output_dir):
    """Q5 supplementary: Generalization experiments from P2.

    Groups K (chunk generalization), L (steps generalization)
    """
    logger.info("=" * 60)
    logger.info("Q5 Supplementary: Generalization")
    logger.info("=" * 60)

    configs = get_configs_by_group("K") + get_configs_by_group("L")
    return run_experiment(engine, configs, systems, output_dir, "q5_supplementary")


def q4_supplementary(engine, systems, output_dir):
    """Q4 supplementary: Diff scaling generalization from P2.

    Group M (Diff S/L on corner VLMs)
    """
    logger.info("=" * 60)
    logger.info("Q4 Supplementary: Diff Scaling")
    logger.info("=" * 60)

    configs = get_configs_by_group("M")
    return run_experiment(engine, configs, systems, output_dir, "q4_supplementary")


EXPERIMENTS = {
    "q1": q1_component_scaling,
    "q2": q2_optimal_allocation,
    "q3": q3_bottleneck,
    "q4": q4_action_architecture,
    "q5": q5_chunk_steps,
    "q6": q6_cross_platform,
    "q5_supp": q5_supplementary,
    "q4_supp": q4_supplementary,
}


def main():
    parser = argparse.ArgumentParser(description="VLA Scaling Experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=list(EXPERIMENTS.keys()) + ["all"],
                        help="Which experiment to run")
    parser.add_argument("--systems", type=str, nargs="+", default=None)
    parser.add_argument("--bits", type=str, default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(output_dir / "scaling_experiments.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    systems = args.systems or DEFAULT_SYSTEMS
    engine = VLAPerfEngine(batch_size=args.batch_size, bits=args.bits)

    if args.experiment == "all":
        for name, func in EXPERIMENTS.items():
            func(engine, systems, args.output_dir)
    else:
        EXPERIMENTS[args.experiment](engine, systems, args.output_dir)


if __name__ == "__main__":
    main()
