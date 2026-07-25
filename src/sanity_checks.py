"""Sanity checks for data, groups, parameter counts, and model invariance."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .data_tictactoe import (
    class_counts,
    generate_all_states,
    labels_are_d4_invariant,
    make_approximate_symmetry_split,
    make_corrupted_labels,
    make_d4_orbits,
    make_balanced_split,
    make_d4_group_holdout_split,
)
from .circuits import LINE_CIRCUIT_FAMILIES, line_parameter_channels, uses_edge_pairs
from .groups_d4 import (
    EXPECTED_PARAMETERS_PER_BLOCK,
    PERMUTATIONS,
    SUBGROUPS,
    apply_permutation_to_board,
    make_sharing_pattern,
    line_orbits,
    pair_orbits,
    qubit_orbits,
)
from .models import QuantumTicTacToeModel, count_parameters
from .rule_based import evaluate_oracle, oracle_is_d4_invariant
from .train import train_model
from .utils import ExperimentConfig, seed_everything


def check_groups_and_orbits() -> None:
    expected_orders = {
        "none": 1,
        "Z2_rot180": 2,
        "Z2_reflection": 2,
        "C4": 4,
        "D2_V4": 4,
        "D4": 8,
    }
    for subgroup, expected_order in expected_orders.items():
        assert len(SUBGROUPS[subgroup]) == expected_order
        sharing = make_sharing_pattern(subgroup)
        assert sharing.n_parameters_per_block == EXPECTED_PARAMETERS_PER_BLOCK[subgroup]
        assert len(qubit_orbits(subgroup)) > 0
        assert len(pair_orbits(subgroup)) > 0
        assert len(line_orbits(subgroup)) > 0

    # Spot-check the vertical reflection described in Meyer et al.
    vertical = PERMUTATIONS["reflect_vertical"]
    assert vertical[0] == 2 and vertical[2] == 0
    assert vertical[7] == 3 and vertical[3] == 7
    assert vertical[6] == 4 and vertical[4] == 6
    assert vertical[8] == 8


def check_dataset() -> None:
    counts = class_counts()
    assert counts == {"circle": 316, "draw": 4536, "cross": 626}
    assert labels_are_d4_invariant()
    oracle_stats = evaluate_oracle()
    assert oracle_stats["accuracy"] == 1.0
    assert oracle_is_d4_invariant()
    split = make_balanced_split(train_size=30, test_size=60, seed=0, allow_overlap_if_needed=False)
    assert split.metadata["actual_disjoint"] is True
    assert len(split.x_train) == 30
    assert len(split.x_test) == 60
    assert set(map(tuple, split.x_train.astype(int))).isdisjoint(
        set(map(tuple, split.x_test.astype(int)))
    )


def check_leak_free_dataset() -> None:
    x, _, _ = generate_all_states()
    board_to_index = {
        tuple(board.astype(int).tolist()): index for index, board in enumerate(x)
    }
    orbit_by_index = {
        state_index: orbit_id
        for orbit_id, orbit in enumerate(make_d4_orbits(x=x))
        for state_index in orbit
    }
    expected_test: set[int] | None = None

    for train_seed in range(10):
        previous_train: set[int] = set()
        for train_size in (30, 60, 120, 240, 450, 600):
            split = make_d4_group_holdout_split(
                train_size=train_size,
                test_size=348,
                train_seed=train_seed,
                test_seed=0,
            )
            train_indices = {
                board_to_index[tuple(board.astype(int).tolist())]
                for board in split.x_train
            }
            test_indices = {
                board_to_index[tuple(board.astype(int).tolist())]
                for board in split.x_test
            }
            train_orbits = {orbit_by_index[index] for index in train_indices}
            test_orbits = {orbit_by_index[index] for index in test_indices}

            assert len(train_indices) == train_size
            assert len(test_indices) == 348
            assert train_indices.isdisjoint(test_indices)
            assert train_orbits.isdisjoint(test_orbits)
            assert previous_train.issubset(train_indices)
            assert split.metadata["actual_disjoint"] is True
            assert split.metadata["group_disjoint"] is True
            assert split.metadata["overlap_count"] == 0
            assert split.metadata["orbit_overlap_count"] == 0
            for class_name in ("circle", "draw", "cross"):
                assert int(np.sum(split.train_labels == class_name)) == train_size // 3
                assert int(np.sum(split.test_labels == class_name)) == 116

            if expected_test is None:
                expected_test = test_indices
            else:
                assert test_indices == expected_test
            previous_train = train_indices


def check_approximate_symmetry_tools() -> None:
    x, _, labels = generate_all_states()
    orbits = make_d4_orbits(x=x, labels=labels)
    assert len(orbits) > 0

    corrupted_labels, meta = make_corrupted_labels(
        x=x,
        labels=labels,
        epsilon=0.2,
        seed=123,
    )
    assert meta["corrupted_orbit_count"] > 0
    changed = int(np.sum(labels != corrupted_labels))
    assert changed > 0
    assert changed <= len(labels)
    # epsilon=0 should be the identity operation.
    identity_labels, identity_meta = make_corrupted_labels(
        x=x,
        labels=labels,
        epsilon=0.0,
        seed=123,
    )
    assert np.array_equal(labels, identity_labels)
    assert identity_meta["requested_epsilon"] == 0.0
    split = make_approximate_symmetry_split(
        train_size=30,
        test_size=60,
        seed=0,
        epsilon=0.2,
        disjoint=True,
        allow_overlap_if_needed=False,
    )
    assert split.metadata["requested_epsilon"] == 0.2
    assert split.metadata["corrupted_orbit_count"] > 0


def _predict_one(model: QuantumTicTacToeModel, board: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(board, dtype=torch.float64)
        return model(x).detach().cpu().numpy()


def check_model_invariance() -> None:
    seed_everything(123)
    board = np.asarray([1, -1, 0, 0, 1, 0, -1, 0, 0], dtype=np.float64)
    for circuit_family in ("edge", *sorted(LINE_CIRCUIT_FAMILIES)):
        for subgroup in ("Z2_rot180", "Z2_reflection", "C4", "D2_V4", "D4"):
            config = ExperimentConfig(
                subgroup=subgroup,
                L=1,
                p=1,
                seed=123,
                circuit_family=circuit_family,
            )
            model = QuantumTicTacToeModel(config)
            base = _predict_one(model, board)
            for transform_name in SUBGROUPS[subgroup]:
                transformed = apply_permutation_to_board(board, PERMUTATIONS[transform_name])
                diff = np.max(np.abs(base - _predict_one(model, transformed)))
                assert diff < 1e-6, (circuit_family, subgroup, transform_name, diff)

    baseline = QuantumTicTacToeModel(ExperimentConfig(subgroup="none", L=1, p=1, seed=123))
    base = _predict_one(baseline, board)
    transformed = apply_permutation_to_board(board, PERMUTATIONS["rot90"])
    diff = np.max(np.abs(base - _predict_one(baseline, transformed)))
    assert diff > 1e-8


def check_parameter_counts() -> None:
    for subgroup, per_block in EXPECTED_PARAMETERS_PER_BLOCK.items():
        ry_config = ExperimentConfig(
            subgroup=subgroup,
            L=3,
            p=2,
            seed=5,
            single_qubit_block="ry",
        )
        ry_model = QuantumTicTacToeModel(ry_config)
        assert count_parameters(ry_config) == 6 * per_block
        assert ry_model.parameter_count == 6 * per_block

        paper_config = ExperimentConfig(
            subgroup=subgroup,
            L=3,
            p=2,
            seed=5,
            single_qubit_block="paper",
        )
        paper_model = QuantumTicTacToeModel(paper_config)
        expected_paper = 6 * (
            2 * paper_model.sharing.n_single_parameters + paper_model.sharing.n_pair_parameters
        )
        assert count_parameters(paper_config) == expected_paper
        assert paper_model.parameter_count == expected_paper

        for circuit_family in sorted(LINE_CIRCUIT_FAMILIES):
            line_config = ExperimentConfig(
                subgroup=subgroup,
                L=3,
                p=2,
                seed=5,
                single_qubit_block="paper",
                circuit_family=circuit_family,
            )
            line_model = QuantumTicTacToeModel(line_config)
            interaction_parameters = (
                line_model.sharing.n_pair_parameters if uses_edge_pairs(circuit_family) else 0
            )
            interaction_parameters += (
                line_parameter_channels(circuit_family) * line_model.sharing.n_line_parameters
            )
            expected_line = 6 * (
                2 * line_model.sharing.n_single_parameters + interaction_parameters
            )
            assert count_parameters(line_config) == expected_line
            assert line_model.parameter_count == expected_line


def run_smoke_training() -> None:
    config = ExperimentConfig(
        subgroup="D4",
        L=1,
        p=1,
        seed=0,
        train_size=30,
        test_size=60,
        batch_size=5,
        epochs=1,
        steps_per_epoch=2,
        lr=0.01,
        allow_overlap_if_needed=False,
    )
    _, row, _ = train_model(config)
    assert 0.0 <= float(row["train_accuracy"]) <= 1.0
    assert 0.0 <= float(row["test_accuracy"]) <= 1.0
    assert int(row["num_parameters"]) == 9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    check_groups_and_orbits()
    check_dataset()
    check_leak_free_dataset()
    check_approximate_symmetry_tools()
    check_model_invariance()
    check_parameter_counts()
    if not args.skip_smoke:
        run_smoke_training()
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
