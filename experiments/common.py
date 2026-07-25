"""Shared CLI helpers for experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd

from src.circuits import ALL_CIRCUIT_FAMILIES
from src.data_tictactoe import DatasetSplit
from src.train import run_and_save_rows, train_model
from src.utils import (
    CSV_DIR,
    ExperimentConfig,
    config_to_dict,
    ensure_results_dirs,
    parse_int_list,
    seed_range,
)

RUN_KEY_FIELDS = (
    "subgroup",
    "L",
    "p",
    "seed",
    "train_size",
    "test_size",
    "batch_size",
    "epochs",
    "steps_per_epoch",
    "lr",
    "random_sharing",
    "pl_device",
    "diff_method",
    "single_qubit_block",
    "circuit_family",
    "epsilon",
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated random seeds.")
    parser.add_argument("--final", action="store_true", help="Use seeds 0..19.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--test-size", type=int, default=600)
    parser.add_argument("--strict-disjoint", action="store_true")
    parser.add_argument("--pl-device", default="lightning.qubit")
    parser.add_argument("--diff-method", default="adjoint")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing CSV and skip configurations already present.",
    )
    parser.add_argument(
        "--single-qubit-block",
        choices=("ry", "paper"),
        default="paper",
        help="'ry' is the low-parameter quick model; 'paper' uses RY RX-style single-qubit gates.",
    )
    parser.add_argument(
        "--circuit-family",
        choices=tuple(sorted(ALL_CIRCUIT_FAMILIES)),
        default="edge",
        help=(
            "'edge' uses CRY gates on directed pairs; line_* families use "
            "trainable 3-qubit winning-line motifs."
        ),
    )


def seeds_from_args(args: argparse.Namespace) -> list[int]:
    return seed_range(20) if args.final else parse_int_list(args.seeds)


def _normalized_key(row: dict) -> tuple:
    key_values = []
    for field in RUN_KEY_FIELDS:
        value = row.get(field)
        if field == "epsilon" and pd.isna(value):
            value = 0.0
        if field in {"L", "p", "seed", "train_size", "test_size", "batch_size", "epochs", "steps_per_epoch"}:
            value = int(value)
        elif field in {"lr", "epsilon"}:
            value = round(float(value), 12)
        elif field == "random_sharing":
            if isinstance(value, str):
                value = value.strip().lower() == "true"
            else:
                value = bool(value)
        key_values.append(value)
    return tuple(key_values)


def _config_key(config: ExperimentConfig) -> tuple:
    return _normalized_key(config_to_dict(config))


def run_configs(
    configs: list[ExperimentConfig],
    output_path: Path,
    *,
    resume: bool = False,
    split_factory: Callable[[ExperimentConfig], DatasetSplit] | None = None,
    resume_row_compatible: Callable[[dict], bool] | None = None,
) -> pd.DataFrame:
    ensure_results_dirs()
    rows: list[dict] = []
    completed_keys: set[tuple] = set()
    if resume and output_path.exists() and output_path.stat().st_size > 0:
        existing = pd.read_csv(output_path)
        rows = existing.to_dict("records")
        if resume_row_compatible is not None:
            incompatible = [row for row in rows if not resume_row_compatible(row)]
            if incompatible:
                raise RuntimeError(
                    f"Refusing to resume {output_path}: {len(incompatible)} rows were "
                    "produced by an incompatible protocol, source tree, or environment."
                )
        completed_keys = {_normalized_key(row) for row in rows}
        print(f"Resuming from {output_path}: {len(rows)} existing rows.", flush=True)

    for index, config in enumerate(configs, start=1):
        if resume and _config_key(config) in completed_keys:
            print(
                f"[{index}/{len(configs)}] skip existing subgroup={config.subgroup} "
                f"L={config.L} p={config.p} seed={config.seed} train={config.train_size} "
                f"epsilon={config.epsilon}",
                flush=True,
            )
            continue
        print(
            f"[{index}/{len(configs)}] subgroup={config.subgroup} "
            f"sharing={'random' if config.random_sharing else 'symmetry'} "
            f"L={config.L} p={config.p} seed={config.seed} train={config.train_size} "
            f"epsilon={config.epsilon}",
            flush=True,
        )
        split = split_factory(config) if split_factory is not None else None
        _, row, _ = train_model(config, split=split)
        rows.append(row)
        completed_keys.add(_config_key(config))
        run_and_save_rows(rows, output_path)
    return pd.DataFrame(rows)


def default_csv_path(filename: str) -> Path:
    return CSV_DIR / filename
