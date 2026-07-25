"""Leak-corrected rerun of the paper matrix with a D4-group holdout.

This module deliberately writes to a separate results namespace and never calls
the paper figure or PDF build. The accepted-paper artifacts remain immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch

from experiments.common import run_configs
from src.data_tictactoe import (
    CLASS_NAMES,
    LEAK_FREE_SPLIT_PROTOCOL,
    DatasetSplit,
    generate_all_states,
    make_balanced_split,
    make_d4_group_holdout_split,
    make_d4_orbits,
)
from src.groups_d4 import subgroup_names
from src.utils import CSV_DIR, PROJECT_ROOT, ExperimentConfig, config_to_dict


TRAIN_SIZES = [30, 60, 120, 240, 450, 600]
EDGE_SUBGROUPS = subgroup_names()
LINE_SUBGROUPS = ["C4", "D4"]
SEEDS = list(range(10))
EDGE_LINES_FAMILY = "edge_line_zzz_ccrz"
ABLATION_FAMILIES = [
    "edge",
    EDGE_LINES_FAMILY,
    "edge_line_zzz",
    "edge_line_ccrz",
    "line_zzz_ccrz",
    "line_zzz",
    "line_ccrz",
    "line_pair_crz",
]

# The accepted paper already fixes 100 epochs. No test-driven budget selection is
# performed in this leak-corrected rerun.
EPOCHS = 100
TEST_SIZE = 348
TEST_SEED = 0
EXPECTED_UNIQUE_RUNS = 650

OUTPUT_DIR = CSV_DIR / "leak_free_d4_v1"
RUNS_CSV = OUTPUT_DIR / "results_leak_free_unique_L3p2.csv"
EDGE_CSV = OUTPUT_DIR / "results_leak_free_edge_L3p2.csv"
EDGE_LINES_CSV = OUTPUT_DIR / "results_leak_free_edge_lines_L3p2.csv"
ABLATION_CSV = OUTPUT_DIR / "results_leak_free_ablation_L3p2_train600.csv"
RANDOM_CSV = OUTPUT_DIR / "results_leak_free_random_sharing_L3p2_train600.csv"
SUMMARY_CSV = OUTPUT_DIR / "table_leak_free_summary.csv"
COMPARISON_CSV = OUTPUT_DIR / "table_legacy_vs_leak_free.csv"
CLAIMS_CSV = OUTPUT_DIR / "table_claims_legacy_vs_leak_free.csv"
MANIFEST_JSON = OUTPUT_DIR / "split_and_run_manifest.json"

SOURCE_PATHS = [
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / "src" / "data_tictactoe.py",
    PROJECT_ROOT / "src" / "train.py",
    PROJECT_ROOT / "src" / "models.py",
    PROJECT_ROOT / "src" / "circuits.py",
    PROJECT_ROOT / "src" / "evaluate.py",
    PROJECT_ROOT / "src" / "groups_d4.py",
    PROJECT_ROOT / "src" / "utils.py",
    PROJECT_ROOT / "experiments" / "common.py",
    PROJECT_ROOT / "experiments" / "leak_free_paper_experiments.py",
    PROJECT_ROOT / "scripts" / "run_leak_free_paper_experiments.sh",
]

PANEL_CONFIGS = {
    "edge": lambda: edge_configs(),
    "edge_lines": lambda: edge_lines_configs(),
    "ablation": lambda: ablation_configs(),
    "random": lambda: random_configs(),
}
PANEL_OUTPUTS = {
    "edge": EDGE_CSV,
    "edge_lines": EDGE_LINES_CSV,
    "ablation": ABLATION_CSV,
    "random": RANDOM_CSV,
}
LEGACY_OUTPUTS = {
    "edge": CSV_DIR / "results_paper_consistent_edge_L3p2.csv",
    "edge_lines": CSV_DIR / "results_paper_consistent_edge_lines_L3p2.csv",
    "ablation": CSV_DIR / "results_paper_consistent_ablation_L3p2_train600.csv",
    "random": CSV_DIR / "results_paper_consistent_random_sharing_L3p2_train600.csv",
}

KEY_FIELDS = (
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
    "split_protocol",
    "test_seed",
)


def _config(
    *,
    subgroup: str,
    seed: int,
    train_size: int,
    circuit_family: str = "edge",
    random_sharing: bool = False,
) -> ExperimentConfig:
    return ExperimentConfig(
        subgroup=subgroup,
        L=3,
        p=2,
        seed=seed,
        train_size=train_size,
        test_size=TEST_SIZE,
        batch_size=15,
        epochs=EPOCHS,
        steps_per_epoch=30,
        lr=0.01,
        random_sharing=random_sharing,
        pl_device="lightning.qubit",
        diff_method="adjoint",
        single_qubit_block="paper",
        circuit_family=circuit_family,
        allow_overlap_if_needed=False,
    )


def edge_configs() -> list[ExperimentConfig]:
    return [
        _config(subgroup=subgroup, seed=seed, train_size=train_size)
        for subgroup in EDGE_SUBGROUPS
        for train_size in TRAIN_SIZES
        for seed in SEEDS
    ]


def edge_lines_configs() -> list[ExperimentConfig]:
    return [
        _config(
            subgroup=subgroup,
            seed=seed,
            train_size=train_size,
            circuit_family=EDGE_LINES_FAMILY,
        )
        for subgroup in LINE_SUBGROUPS
        for train_size in TRAIN_SIZES
        for seed in SEEDS
    ]


def ablation_configs() -> list[ExperimentConfig]:
    return [
        _config(subgroup=subgroup, seed=seed, train_size=600, circuit_family=family)
        for family in ABLATION_FAMILIES
        for subgroup in LINE_SUBGROUPS
        for seed in SEEDS
    ]


def random_configs() -> list[ExperimentConfig]:
    configs = [_config(subgroup="none", seed=seed, train_size=600) for seed in SEEDS]
    for subgroup in ["Z2_reflection", "Z2_rot180", "C4", "D2_V4", "D4"]:
        for random_sharing in [False, True]:
            for seed in SEEDS:
                configs.append(
                    _config(
                        subgroup=subgroup,
                        seed=seed,
                        train_size=600,
                        random_sharing=random_sharing,
                    )
                )
    return configs


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _row_key(row: dict | pd.Series) -> tuple:
    values: list[object] = []
    for field in KEY_FIELDS:
        value = row.get(field)
        if field in {
            "L",
            "p",
            "seed",
            "train_size",
            "test_size",
            "batch_size",
            "epochs",
            "steps_per_epoch",
            "test_seed",
        }:
            value = int(value)
        elif field in {"lr", "epsilon"}:
            value = round(float(0.0 if pd.isna(value) else value), 12)
        elif field == "random_sharing":
            value = _as_bool(value)
        else:
            value = str(value)
        values.append(value)
    return tuple(values)


def _config_key(config: ExperimentConfig) -> tuple:
    row = config_to_dict(config)
    row.update({"split_protocol": LEAK_FREE_SPLIT_PROTOCOL, "test_seed": TEST_SEED})
    return _row_key(row)


def unique_configs() -> list[ExperimentConfig]:
    configs: list[ExperimentConfig] = []
    seen: set[tuple] = set()
    for panel_configs in (
        edge_configs(),
        edge_lines_configs(),
        ablation_configs(),
        random_configs(),
    ):
        for config in panel_configs:
            key = _config_key(config)
            if key not in seen:
                configs.append(config)
                seen.add(key)
    if len(configs) != EXPECTED_UNIQUE_RUNS:
        raise RuntimeError(
            f"Experiment matrix changed: expected {EXPECTED_UNIQUE_RUNS} unique runs, got {len(configs)}."
        )
    return configs


def _split_for_config(config: ExperimentConfig) -> DatasetSplit:
    split = make_d4_group_holdout_split(
        train_size=config.train_size,
        test_size=config.test_size,
        train_seed=config.seed,
        test_seed=TEST_SEED,
    )
    split.metadata.update(
        {
            "source_fingerprint": _source_fingerprint(),
            "environment_fingerprint": _environment_fingerprint(),
            "run_fingerprint": _run_fingerprint(),
        }
    )
    return split


def _resume_row_compatible(row: dict) -> bool:
    return str(row.get("run_fingerprint", "")) == _run_fingerprint()


def _shard_path(path: Path, shard_index: int, shard_count: int) -> Path:
    return path.with_name(
        f"{path.stem}.shard{shard_index:02d}of{shard_count:02d}{path.suffix}"
    )


def _dedupe(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["_key"] = df.apply(_row_key, axis=1)
    return df.drop_duplicates("_key", keep="last").drop(columns="_key")


def _read_runs(*, shard_count: int | None = None) -> pd.DataFrame:
    shard_pattern = re.compile(r"\.shard\d+of(\d+)\.csv$")
    all_shards = sorted(OUTPUT_DIR.glob(f"{RUNS_CSV.stem}.shard*of*{RUNS_CSV.suffix}"))
    counts = {
        int(match.group(1))
        for path in all_shards
        if (match := shard_pattern.search(path.name)) is not None
    }
    if shard_count is None:
        if len(counts) > 1:
            raise RuntimeError(
                f"Found incompatible shard-count sets {sorted(counts)}; "
                "pass --shard-count explicitly."
            )
        shard_count = next(iter(counts), 1)
    selected_shards = [
        path
        for path in all_shards
        if (match := shard_pattern.search(path.name)) is not None
        and int(match.group(1)) == shard_count
    ]
    candidates = [RUNS_CSV, *selected_shards] if shard_count > 1 else [RUNS_CSV]
    frames = [
        pd.read_csv(path)
        for path in candidates
        if path.exists() and path.stat().st_size > 0
    ]
    return _dedupe(frames)


def run_unique(
    *,
    resume: bool = True,
    shard_index: int | None = None,
    shard_count: int = 1,
) -> pd.DataFrame:
    configs = unique_configs()
    if shard_count == 1:
        return run_configs(
            configs,
            RUNS_CSV,
            resume=resume,
            split_factory=_split_for_config,
            resume_row_compatible=_resume_row_compatible,
        )
    if shard_index is None or not 0 <= shard_index < shard_count:
        raise ValueError(
            "--shard-index must be in [0, shard_count) when --shard-count > 1."
        )

    # Assign before inspecting partial outputs so concurrent shard startup cannot
    # shift indices and silently omit configurations.
    shard_configs = [
        config
        for index, config in enumerate(configs)
        if index % shard_count == shard_index
    ]
    shard_path = _shard_path(RUNS_CSV, shard_index, shard_count)
    print(
        f"Shard {shard_index}/{shard_count}: {len(shard_configs)} configs -> {shard_path}"
    )
    return run_configs(
        shard_configs,
        shard_path,
        resume=resume,
        split_factory=_split_for_config,
        resume_row_compatible=_resume_row_compatible,
    )


def merge_runs(*, shard_count: int | None = None) -> pd.DataFrame:
    df = _read_runs(shard_count=shard_count)
    if df.empty:
        raise RuntimeError("No leak-free run outputs found.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUNS_CSV, index=False)
    print(f"Merged {len(df)}/{EXPECTED_UNIQUE_RUNS} unique runs -> {RUNS_CSV}")
    return df


def materialize_panels(*, shard_count: int | None = None) -> None:
    runs = _read_runs(shard_count=shard_count)
    if len(runs) != EXPECTED_UNIQUE_RUNS:
        raise RuntimeError(
            f"Cannot materialize panels: expected {EXPECTED_UNIQUE_RUNS} unique runs, got {len(runs)}."
        )
    row_by_key = {_row_key(row): row.to_dict() for _, row in runs.iterrows()}
    summary_frames: list[pd.DataFrame] = []

    for panel, config_factory in PANEL_CONFIGS.items():
        rows = [row_by_key[_config_key(config)] for config in config_factory()]
        panel_df = pd.DataFrame(rows)
        panel_path = PANEL_OUTPUTS[panel]
        panel_df.to_csv(panel_path, index=False)
        grouped = (
            panel_df.groupby(
                ["circuit_family", "subgroup", "train_size", "random_sharing"],
                dropna=False,
            )
            .agg(
                n=("seed", "nunique"),
                test_accuracy=("test_accuracy", "mean"),
                train_accuracy=("train_accuracy", "mean"),
                generalization_gap=("generalization_gap", "mean"),
                num_parameters=("num_parameters", "mean"),
            )
            .reset_index()
        )
        grouped["source"] = panel_path.name
        summary_frames.append(grouped)
        print(f"Materialized {len(panel_df)} rows -> {panel_path}")

    pd.concat(summary_frames, ignore_index=True).to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {SUMMARY_CSV}")


def _indices_from_split(
    split: DatasetSplit, board_to_index: dict[tuple[int, ...], int]
) -> list[int]:
    return sorted(
        board_to_index[tuple(board.astype(int).tolist())] for board in split.x_train
    )


def _test_indices_from_split(
    split: DatasetSplit, board_to_index: dict[tuple[int, ...], int]
) -> list[int]:
    return sorted(
        board_to_index[tuple(board.astype(int).tolist())] for board in split.x_test
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _environment_info() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "pennylane": qml.__version__,
    }


@lru_cache(maxsize=1)
def _environment_fingerprint() -> str:
    payload = json.dumps(_environment_info(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _run_fingerprint() -> str:
    payload = (
        f"{LEAK_FREE_SPLIT_PROTOCOL}:{_source_fingerprint()}:"
        f"{_environment_fingerprint()}"
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _indices_digest(indices: set[int] | list[int]) -> str:
    payload = ",".join(str(index) for index in sorted(int(index) for index in indices))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@lru_cache(maxsize=None)
def _legacy_orbit_leak(seed: int, train_size: int) -> tuple[int, int]:
    x, _, _ = generate_all_states()
    board_to_index = {
        tuple(board.astype(int).tolist()): index for index, board in enumerate(x)
    }
    orbit_by_index = {
        state_index: orbit_id
        for orbit_id, orbit in enumerate(make_d4_orbits(x=x))
        for state_index in orbit
    }
    split = make_balanced_split(
        train_size=train_size,
        test_size=600,
        seed=seed,
        disjoint=True,
        allow_overlap_if_needed=True,
    )
    train_indices = {
        board_to_index[tuple(board.astype(int).tolist())] for board in split.x_train
    }
    test_indices = {
        board_to_index[tuple(board.astype(int).tolist())] for board in split.x_test
    }
    train_orbits = {orbit_by_index[index] for index in train_indices}
    test_orbits = {orbit_by_index[index] for index in test_indices}
    shared_orbits = train_orbits & test_orbits
    affected_test_states = sum(
        1 for index in test_indices if orbit_by_index[index] in shared_orbits
    )
    return len(shared_orbits), affected_test_states


def write_manifest() -> None:
    x, _, labels = generate_all_states()
    board_to_index = {
        tuple(board.astype(int).tolist()): index for index, board in enumerate(x)
    }
    d4_orbits = make_d4_orbits(x=x)
    orbit_by_index = {
        state_index: orbit_id
        for orbit_id, orbit in enumerate(d4_orbits)
        for state_index in orbit
    }

    split_rows: list[dict[str, object]] = []
    fixed_test_indices: list[int] | None = None
    for seed in SEEDS:
        previous_train: set[int] = set()
        for train_size in TRAIN_SIZES:
            split = make_d4_group_holdout_split(
                train_size=train_size,
                test_size=TEST_SIZE,
                train_seed=seed,
                test_seed=TEST_SEED,
            )
            train_indices = _indices_from_split(split, board_to_index)
            test_indices = _test_indices_from_split(split, board_to_index)
            if fixed_test_indices is None:
                fixed_test_indices = test_indices
            elif test_indices != fixed_test_indices:
                raise RuntimeError(
                    "Leak-free protocol must use one fixed test set for every run."
                )
            if not previous_train.issubset(train_indices):
                raise RuntimeError(
                    f"Training sets are not nested for seed={seed}, size={train_size}."
                )
            previous_train = set(train_indices)
            split_rows.append(
                {
                    "seed": seed,
                    "train_size": train_size,
                    "train_indices": train_indices,
                    "train_orbit_ids": sorted(
                        {orbit_by_index[index] for index in train_indices}
                    ),
                    "train_index_digest": split.metadata["train_index_digest"],
                    "split_index_digest": split.metadata["split_index_digest"],
                    "class_counts": {
                        class_name: int(np.sum(split.train_labels == class_name))
                        for class_name in CLASS_NAMES
                    },
                }
            )

    assert fixed_test_indices is not None
    fixed_split = make_d4_group_holdout_split(
        train_size=600,
        test_size=TEST_SIZE,
        train_seed=0,
        test_seed=TEST_SEED,
    )
    pdf_paths = [
        PROJECT_ROOT / "paper" / "fig1_4panel_standalone.pdf",
        PROJECT_ROOT / "paper" / "main.pdf",
        PROJECT_ROOT / "paper" / "main_anonymous.pdf",
        PROJECT_ROOT / "paper" / "gfx" / "fig2_main_evidence.pdf",
        PROJECT_ROOT / "paper" / "gfx" / "fig3_controls.pdf",
    ]
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    manifest = {
        "protocol": LEAK_FREE_SPLIT_PROTOCOL,
        "rerun_design": {
            "changed_from_accepted_paper": [
                "fixed D4-orbit-disjoint holdout",
                "test_size reduced from 600 to the maximum feasible balanced 348",
                "overlap fallback disabled",
            ],
            "held_fixed": {
                "train_sizes": TRAIN_SIZES,
                "seeds": SEEDS,
                "L": 3,
                "p": 2,
                "epochs": EPOCHS,
                "steps_per_epoch": 30,
                "batch_size": 15,
                "optimizer": "Adam",
                "learning_rate": 0.01,
                "device": "lightning.qubit",
                "diff_method": "adjoint",
                "single_qubit_block": "paper",
                "model_matrix": "identical to the four final paper panels",
            },
            "selection_policy": (
                "No tuning or selection is performed within this rerun; 100 epochs is copied "
                "from the accepted-paper final protocol before leak-free results are generated."
            ),
            "historical_holdout_caveat": (
                "The holdout comes from the same finite enumerated board corpus as the accepted "
                "paper and is not claimed to be historically unseen by the researchers. No old "
                "model, checkpoint, or per-board prediction is reused."
            ),
            "unique_runs": EXPECTED_UNIQUE_RUNS,
            "materialized_panel_rows": sum(
                len(factory()) for factory in PANEL_CONFIGS.values()
            ),
        },
        "fixed_test_split": {
            "test_seed": TEST_SEED,
            "test_size": TEST_SIZE,
            "test_indices": fixed_test_indices,
            "test_orbit_ids": sorted(
                {orbit_by_index[index] for index in fixed_test_indices}
            ),
            "test_index_digest": fixed_split.metadata["test_index_digest"],
            "class_counts": {
                class_name: int(np.sum(fixed_split.test_labels == class_name))
                for class_name in CLASS_NAMES
            },
        },
        "training_splits": split_rows,
        "environment": {
            "source_git_commit": git_commit,
            "source_file_sha256": {
                str(path.relative_to(PROJECT_ROOT)): _file_sha256(path)
                for path in SOURCE_PATHS
            },
            "source_fingerprint": _source_fingerprint(),
            "environment_fingerprint": _environment_fingerprint(),
            "run_fingerprint": _run_fingerprint(),
            **_environment_info(),
        },
        "immutable_pdf_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _file_sha256(path)
            for path in pdf_paths
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST_JSON}")


def _validate_panel(
    panel: str, expected_configs: list[ExperimentConfig]
) -> pd.DataFrame:
    path = PANEL_OUTPUTS[panel]
    if not path.exists():
        raise RuntimeError(f"Missing leak-free panel output: {path}")
    df = pd.read_csv(path)
    expected_keys = [_config_key(config) for config in expected_configs]
    actual_keys = [_row_key(row) for _, row in df.iterrows()]
    if len(actual_keys) != len(expected_keys) or set(actual_keys) != set(expected_keys):
        raise RuntimeError(
            f"{path} does not match its frozen matrix: expected {len(expected_keys)} rows, "
            f"got {len(actual_keys)}."
        )
    return df


def validate(*, shard_count: int | None = None) -> None:
    runs = _read_runs(shard_count=shard_count)
    if len(runs) != EXPECTED_UNIQUE_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_UNIQUE_RUNS} unique runs, got {len(runs)}."
        )

    required_values = {
        "epochs": EPOCHS,
        "steps_per_epoch": 30,
        "batch_size": 15,
        "test_size": TEST_SIZE,
        "test_seed": TEST_SEED,
        "split_protocol": LEAK_FREE_SPLIT_PROTOCOL,
        "split_group": "D4",
    }
    for column, expected in required_values.items():
        values = runs[column].drop_duplicates().tolist()
        if values != [expected]:
            raise RuntimeError(
                f"Unexpected {column}: {values}; expected only {expected!r}."
            )
    for column in ("actual_disjoint", "group_disjoint"):
        if not runs[column].map(_as_bool).all():
            raise RuntimeError(f"{column} is not true for every leak-free run.")
    for column in ("overlap_count", "orbit_overlap_count"):
        if not (runs[column].astype(int) == 0).all():
            raise RuntimeError(f"{column} is nonzero in leak-free outputs.")
    if runs["test_index_digest"].nunique() != 1:
        raise RuntimeError("The leak-free run did not use one fixed test set.")
    if not (runs["allow_overlap_if_needed"].map(_as_bool) == False).all():  # noqa: E712
        raise RuntimeError("Overlap fallback was enabled in a leak-free run.")
    for column, expected in {
        "source_fingerprint": _source_fingerprint(),
        "environment_fingerprint": _environment_fingerprint(),
        "run_fingerprint": _run_fingerprint(),
    }.items():
        values = runs[column].astype(str).drop_duplicates().tolist()
        if values != [expected]:
            raise RuntimeError(f"Mixed or stale {column} values: {values}.")

    finite_columns = [
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
        "generalization_gap",
        "num_parameters",
        "elapsed_seconds",
    ]
    finite_values = runs[finite_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(finite_values.to_numpy(dtype=float)).all():
        raise RuntimeError("Leak-free outputs contain NaN or infinite metrics.")
    for column in ("train_accuracy", "test_accuracy"):
        if not runs[column].astype(float).between(0.0, 1.0).all():
            raise RuntimeError(f"{column} is outside [0, 1].")
    for column in ("train_loss", "test_loss", "elapsed_seconds", "num_parameters"):
        if not (runs[column].astype(float) > 0.0).all():
            raise RuntimeError(f"{column} must be strictly positive.")
    if not runs["generalization_gap"].astype(float).between(-1.0, 1.0).all():
        raise RuntimeError("generalization_gap is outside [-1, 1].")

    for panel, config_factory in PANEL_CONFIGS.items():
        _validate_panel(panel, config_factory())
    if not MANIFEST_JSON.exists():
        raise RuntimeError(f"Missing split manifest: {MANIFEST_JSON}")

    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    expected_test_digest: str | None = None
    expected_split_digests: dict[tuple[int, int], str] = {}
    manifest_train_indices: dict[tuple[int, int], set[int]] = {}
    manifest_train_orbits: dict[tuple[int, int], set[int]] = {}
    previous_train_by_seed: dict[int, set[int]] = {seed: set() for seed in SEEDS}
    for item in manifest["training_splits"]:
        seed = int(item["seed"])
        train_size = int(item["train_size"])
        train_indices = set(int(index) for index in item["train_indices"])
        if len(train_indices) != train_size:
            raise RuntimeError(
                f"Manifest has duplicate/missing train indices for {(seed, train_size)}."
            )
        if not previous_train_by_seed[seed].issubset(train_indices):
            raise RuntimeError(
                f"Manifest train sets are not nested for {(seed, train_size)}."
            )
        previous_train_by_seed[seed] = train_indices
        if _indices_digest(train_indices) != str(item["train_index_digest"]):
            raise RuntimeError(
                f"Manifest train digest mismatch for {(seed, train_size)}."
            )
        manifest_train_indices[(seed, train_size)] = train_indices
        manifest_train_orbits[(seed, train_size)] = set(
            int(index) for index in item["train_orbit_ids"]
        )
        expected_split_digests[(seed, train_size)] = str(item["split_index_digest"])

    fixed_test_indices = set(
        int(index) for index in manifest["fixed_test_split"]["test_indices"]
    )
    if len(fixed_test_indices) != TEST_SIZE:
        raise RuntimeError("Manifest fixed test set has duplicate or missing indices.")
    manifest_test_digest = str(manifest["fixed_test_split"]["test_index_digest"])
    fixed_test_orbits = set(
        int(index) for index in manifest["fixed_test_split"]["test_orbit_ids"]
    )
    if _indices_digest(fixed_test_indices) != manifest_test_digest:
        raise RuntimeError("Manifest test indices do not match their recorded digest.")
    for key, train_indices in manifest_train_indices.items():
        if not train_indices.isdisjoint(fixed_test_indices):
            raise RuntimeError(f"Manifest has exact train/test overlap for {key}.")
        train_digest = _indices_digest(train_indices)
        combined_digest = hashlib.sha256(
            f"{train_digest}:{manifest_test_digest}".encode("ascii")
        ).hexdigest()
        if combined_digest != expected_split_digests[key]:
            raise RuntimeError(f"Manifest combined split digest mismatch for {key}.")
        if not manifest_train_orbits[key].isdisjoint(fixed_test_orbits):
            raise RuntimeError(f"Manifest has D4-orbit train/test overlap for {key}.")

    expected_seed_sizes = {
        (seed, train_size) for seed in SEEDS for train_size in TRAIN_SIZES
    }
    if set(expected_split_digests) != expected_seed_sizes:
        raise RuntimeError(
            "Manifest does not contain exactly the frozen 10x6 training splits."
        )

    for (seed, train_size), block in runs.groupby(["seed", "train_size"]):
        split = make_d4_group_holdout_split(
            train_size=int(train_size),
            test_size=TEST_SIZE,
            train_seed=int(seed),
            test_seed=TEST_SEED,
        )
        recomputed_test_digest = str(split.metadata["test_index_digest"])
        recomputed_split_digest = str(split.metadata["split_index_digest"])
        if expected_test_digest is None:
            expected_test_digest = recomputed_test_digest
        elif recomputed_test_digest != expected_test_digest:
            raise RuntimeError("Recomputed test digest changed across configurations.")
        if (
            block["test_index_digest"].astype(str).nunique() != 1
            or str(block["test_index_digest"].iloc[0]) != recomputed_test_digest
        ):
            raise RuntimeError(
                f"Stale or mixed test split rows for {(seed, train_size)}."
            )
        if (
            block["split_index_digest"].astype(str).nunique() != 1
            or str(block["split_index_digest"].iloc[0]) != recomputed_split_digest
        ):
            raise RuntimeError(
                f"Stale or mixed train split rows for {(seed, train_size)}."
            )
        if (
            expected_split_digests.get((int(seed), int(train_size)))
            != recomputed_split_digest
        ):
            raise RuntimeError(
                f"Manifest split digest mismatch for {(seed, train_size)}."
            )

    if expected_test_digest != manifest_test_digest:
        raise RuntimeError(
            "Manifest test digest does not match the recomputed fixed holdout."
        )

    print(
        json.dumps(
            {
                "status": "valid",
                "protocol": LEAK_FREE_SPLIT_PROTOCOL,
                "unique_runs": len(runs),
                "test_size": TEST_SIZE,
                "test_index_digest": runs["test_index_digest"].iloc[0],
                "exact_overlap_rows": int(
                    (runs["overlap_count"].astype(int) != 0).sum()
                ),
                "orbit_overlap_rows": int(
                    (runs["orbit_overlap_count"].astype(int) != 0).sum()
                ),
            },
            indent=2,
        )
    )


def compare_with_legacy() -> None:
    group_fields = ["circuit_family", "subgroup", "train_size", "random_sharing"]
    join_fields = [*group_fields, "seed"]
    summaries: list[pd.DataFrame] = []

    for panel in PANEL_OUTPUTS:
        legacy = pd.read_csv(LEGACY_OUTPUTS[panel])
        leak_free = pd.read_csv(PANEL_OUTPUTS[panel])
        legacy_leaks = [
            _legacy_orbit_leak(int(row.seed), int(row.train_size))
            for row in legacy.itertuples(index=False)
        ]
        legacy["orbit_overlap_count"] = [item[0] for item in legacy_leaks]
        legacy["test_states_in_train_orbits"] = [item[1] for item in legacy_leaks]
        paired = legacy.merge(
            leak_free,
            on=join_fields,
            suffixes=("_legacy", "_leak_free"),
            validate="one_to_one",
        )
        if len(paired) != len(legacy) or len(paired) != len(leak_free):
            raise RuntimeError(
                f"Legacy/leak-free key mismatch in panel {panel}: "
                f"legacy={len(legacy)}, leak_free={len(leak_free)}, paired={len(paired)}."
            )
        paired["delta_test_accuracy"] = (
            paired["test_accuracy_leak_free"] - paired["test_accuracy_legacy"]
        )
        paired["delta_train_accuracy"] = (
            paired["train_accuracy_leak_free"] - paired["train_accuracy_legacy"]
        )
        paired["delta_generalization_gap"] = (
            paired["generalization_gap_leak_free"] - paired["generalization_gap_legacy"]
        )
        summary = (
            paired.groupby(group_fields, dropna=False)
            .agg(
                n=("seed", "nunique"),
                legacy_test_accuracy=("test_accuracy_legacy", "mean"),
                leak_free_test_accuracy=("test_accuracy_leak_free", "mean"),
                delta_test_accuracy=("delta_test_accuracy", "mean"),
                delta_test_accuracy_sd=("delta_test_accuracy", "std"),
                legacy_train_accuracy=("train_accuracy_legacy", "mean"),
                leak_free_train_accuracy=("train_accuracy_leak_free", "mean"),
                delta_train_accuracy=("delta_train_accuracy", "mean"),
                legacy_generalization_gap=("generalization_gap_legacy", "mean"),
                leak_free_generalization_gap=("generalization_gap_leak_free", "mean"),
                delta_generalization_gap=("delta_generalization_gap", "mean"),
                legacy_exact_overlap=("overlap_count_legacy", "mean"),
                leak_free_exact_overlap=("overlap_count_leak_free", "mean"),
                legacy_orbit_overlap=("orbit_overlap_count_legacy", "mean"),
                leak_free_orbit_overlap=("orbit_overlap_count_leak_free", "mean"),
                legacy_test_states_in_train_orbits=(
                    "test_states_in_train_orbits",
                    "mean",
                ),
            )
            .reset_index()
        )
        summary["delta_test_accuracy_ci95"] = (
            1.96 * summary["delta_test_accuracy_sd"] / np.sqrt(summary["n"])
        )
        summary.insert(0, "panel", panel)
        summary["legacy_test_size"] = 600
        summary["leak_free_test_size"] = TEST_SIZE
        summaries.append(summary)

    comparison = pd.concat(summaries, ignore_index=True)
    comparison.to_csv(COMPARISON_CSV, index=False)

    legacy_edge = pd.read_csv(LEGACY_OUTPUTS["edge"])
    new_edge = pd.read_csv(EDGE_CSV)
    legacy_lines = pd.read_csv(LEGACY_OUTPUTS["edge_lines"])
    new_lines = pd.read_csv(EDGE_LINES_CSV)
    legacy_random = pd.read_csv(LEGACY_OUTPUTS["random"])
    new_random = pd.read_csv(RANDOM_CSV)

    def mean_accuracy(df: pd.DataFrame, *, family: str, subgroup: str) -> float:
        selected = df[
            (df["circuit_family"] == family)
            & (df["subgroup"] == subgroup)
            & (df["train_size"] == 600)
        ]
        return float(selected["test_accuracy"].mean())

    def random_control_gain(df: pd.DataFrame) -> float:
        selected = df[df["subgroup"] != "none"]
        symmetry = selected[selected["random_sharing"].map(_as_bool) == False][  # noqa: E712
            "test_accuracy"
        ].mean()
        random = selected[selected["random_sharing"].map(_as_bool)][
            "test_accuracy"
        ].mean()
        return float(symmetry - random)

    legacy_none = mean_accuracy(legacy_edge, family="edge", subgroup="none")
    new_none = mean_accuracy(new_edge, family="edge", subgroup="none")
    legacy_d4 = mean_accuracy(legacy_edge, family="edge", subgroup="D4")
    new_d4 = mean_accuracy(new_edge, family="edge", subgroup="D4")
    legacy_lines_d4 = mean_accuracy(
        legacy_lines, family=EDGE_LINES_FAMILY, subgroup="D4"
    )
    new_lines_d4 = mean_accuracy(new_lines, family=EDGE_LINES_FAMILY, subgroup="D4")
    claim_rows = [
        ("edge/none test accuracy at train=600", legacy_none, new_none),
        ("edge/D4 test accuracy at train=600", legacy_d4, new_d4),
        ("edge+lines/D4 test accuracy at train=600", legacy_lines_d4, new_lines_d4),
        (
            "D4 symmetry gain over none at train=600",
            legacy_d4 - legacy_none,
            new_d4 - new_none,
        ),
        (
            "line-interaction gain over edge/D4 at train=600",
            legacy_lines_d4 - legacy_d4,
            new_lines_d4 - new_d4,
        ),
        (
            "orbit-sharing gain over random sharing at train=600",
            random_control_gain(legacy_random),
            random_control_gain(new_random),
        ),
    ]
    claims = pd.DataFrame(claim_rows, columns=["quantity", "legacy", "leak_free"])
    claims["delta"] = claims["leak_free"] - claims["legacy"]
    claims.to_csv(CLAIMS_CSV, index=False)
    print(f"Wrote {COMPARISON_CSV}")
    print(f"Wrote {CLAIMS_CSV}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "run",
            "merge",
            "materialize",
            "manifest",
            "validate",
            "compare",
            "finalize",
        ],
        required=True,
    )
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "run":
        run_unique(
            resume=not args.no_resume,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    elif args.stage == "merge":
        merge_runs(shard_count=args.shard_count)
    elif args.stage == "materialize":
        materialize_panels(shard_count=args.shard_count)
    elif args.stage == "manifest":
        write_manifest()
    elif args.stage == "validate":
        validate(shard_count=args.shard_count)
    elif args.stage == "compare":
        compare_with_legacy()
    elif args.stage == "finalize":
        merge_runs(shard_count=args.shard_count)
        materialize_panels(shard_count=args.shard_count)
        write_manifest()
        validate(shard_count=args.shard_count)
        compare_with_legacy()


if __name__ == "__main__":
    main()
