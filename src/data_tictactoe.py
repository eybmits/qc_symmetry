"""Legal Tic-Tac-Toe board generation and reproducible balanced splits."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from typing import Iterable

import numpy as np

from .groups_d4 import PERMUTATIONS, SUBGROUPS, apply_permutation_to_board, subgroup_permutations

WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (7, 8, 3),
    (6, 5, 4),
    (0, 7, 6),
    (1, 8, 5),
    (2, 3, 4),
    (0, 8, 4),
    (2, 8, 6),
)

CLASS_NAMES = ("circle", "draw", "cross")
LABEL_VECTORS: dict[str, tuple[int, int, int]] = {
    "circle": (1, -1, -1),
    "draw": (-1, 1, -1),
    "cross": (-1, -1, 1),
}

LEAK_FREE_SPLIT_PROTOCOL = "d4_orbit_holdout_v1"


def winner(board: Iterable[int]) -> int:
    """Return +1 for cross/X win, -1 for circle/O win, and 0 otherwise."""
    values = tuple(board)
    for line in WIN_LINES:
        line_sum = sum(values[i] for i in line)
        if line_sum == 3:
            return 1
        if line_sum == -3:
            return -1
    return 0


def label_name(board: Iterable[int]) -> str:
    result = winner(board)
    if result == -1:
        return "circle"
    if result == 1:
        return "cross"
    return "draw"


def label_vector(name: str) -> np.ndarray:
    return np.asarray(LABEL_VECTORS[name], dtype=np.float64)


@lru_cache(maxsize=1)
def _generated_states() -> tuple[tuple[int, ...], ...]:
    states: set[tuple[int, ...]] = set()
    expanded: set[tuple[int, ...]] = set()

    def walk(board: tuple[int, ...]) -> None:
        states.add(board)
        if board in expanded:
            return
        expanded.add(board)
        if winner(board) != 0 or all(value != 0 for value in board):
            return
        n_cross = board.count(1)
        n_circle = board.count(-1)
        player = 1 if n_cross == n_circle else -1
        for idx, value in enumerate(board):
            if value == 0:
                next_board = list(board)
                next_board[idx] = player
                walk(tuple(next_board))

    walk((0,) * 9)
    return tuple(sorted(states))


def generate_all_states() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all unique legal board states, target vectors, and class names."""
    boards = _generated_states()
    labels = np.asarray([label_name(board) for board in boards], dtype=object)
    x = np.asarray(boards, dtype=np.float64)
    y = np.asarray([label_vector(name) for name in labels], dtype=np.float64)
    return x, y, labels


def make_subgroup_orbits(
    subgroup: str = "D4",
    *,
    x: np.ndarray | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return index orbits of legal boards under a subgroup of D4."""
    if subgroup not in SUBGROUPS:
        raise KeyError(f"Unknown subgroup {subgroup!r}. Options: {sorted(SUBGROUPS)}")
    if x is None:
        x, _, _ = generate_all_states()
    perms = subgroup_permutations(subgroup)
    board_to_index = {tuple(board.astype(int).tolist()): idx for idx, board in enumerate(x)}
    remaining = set(range(len(x)))
    orbits: list[tuple[int, ...]] = []
    while remaining:
        seed_index = min(remaining)
        board = x[seed_index]
        orbit_indices: set[int] = set()
        for perm in perms:
            transformed = apply_permutation_to_board(board.astype(int), perm)
            orbit_indices.add(board_to_index[tuple(transformed.astype(int).tolist())])
        orbit = tuple(sorted(orbit_indices))
        orbits.append(orbit)
        remaining.difference_update(orbit_indices)
    return tuple(sorted(orbits))


def make_d4_orbits(
    *,
    x: np.ndarray | None = None,
    labels: np.ndarray | None = None,  # kept for backwards-compatible signature
) -> tuple[tuple[int, ...], ...]:
    """Return index orbits of legal boards under all D4 transforms."""
    del labels
    return make_subgroup_orbits("D4", x=x)


def class_counts(labels: np.ndarray | None = None) -> dict[str, int]:
    if labels is None:
        _, _, labels = generate_all_states()
    return {name: int(np.sum(labels == name)) for name in CLASS_NAMES}


@dataclass(frozen=True)
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    train_labels: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_labels: np.ndarray
    metadata: dict[str, int | bool | str | float | tuple[int, ...]]


def make_corrupted_labels(
    *,
    x: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    seed: int = 0,
    corruption_mode: str = "swap_circle_cross",
    broken_K: str = "C4",
) -> tuple[np.ndarray, dict[str, int | bool | float | str | tuple[int, ...]]]:
    """Return labels with K-orbit-based corruption that breaks D4 down to ``broken_K``.

    For ``epsilon > 0`` we pick a fraction ``epsilon`` of the orbits of the
    subgroup ``K = broken_K`` over the legal board states and swap circle ↔
    cross labels uniformly within each selected K-orbit. By construction the
    resulting labels are K-invariant but no longer D4-invariant whenever
    ``K ⊊ D4``, so a model with `H ⊆ K` has the correct inductive bias while a
    model with `H ⊄ K` is over-constrained.
    """
    labels = np.asarray(labels, dtype=object)
    if epsilon <= 0.0:
        return labels.copy(), {
            "requested_epsilon": 0.0,
            "actual_epsilon": 0.0,
            "corrupted_orbit_count": 0,
            "corruption_mode": corruption_mode,
            "broken_K": broken_K,
            "corrupted_orbit_indices": tuple(),
        }

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    if corruption_mode != "swap_circle_cross":
        raise ValueError(f"Unsupported corruption_mode={corruption_mode!r}")
    if broken_K not in SUBGROUPS:
        raise KeyError(f"Unknown broken_K {broken_K!r}. Options: {sorted(SUBGROUPS)}")

    orbits = make_subgroup_orbits(broken_K, x=x)
    n_orbits = len(orbits)
    target_count = int(round(epsilon * n_orbits))
    target_count = max(0, min(n_orbits, target_count))

    rng = np.random.default_rng(seed)
    orbit_order = np.arange(n_orbits)
    rng.shuffle(orbit_order)
    selected_orbit_indices = tuple(int(idx) for idx in orbit_order[:target_count])

    corrupted = labels.copy()
    for orbit_idx in selected_orbit_indices:
        for state_idx in orbits[orbit_idx]:
            current = str(corrupted[state_idx])
            if current == "circle":
                corrupted[state_idx] = "cross"
            elif current == "cross":
                corrupted[state_idx] = "circle"

    return corrupted, {
        "requested_epsilon": float(epsilon),
        "actual_epsilon": float(target_count / n_orbits if n_orbits else 0.0),
        "corrupted_orbit_count": len(selected_orbit_indices),
        "corruption_mode": corruption_mode,
        "broken_K": broken_K,
        "corrupted_orbit_indices": selected_orbit_indices,
    }


def _check_balanced_size(size: int, name: str) -> int:
    if size % len(CLASS_NAMES) != 0:
        raise ValueError(f"{name}={size} must be divisible by {len(CLASS_NAMES)} classes")
    return size // len(CLASS_NAMES)


def _index_set_digest(indices: Iterable[int]) -> str:
    payload = ",".join(str(index) for index in sorted(int(index) for index in indices))
    return sha256(payload.encode("ascii")).hexdigest()


def _choose_orbits_with_exact_size(
    orbits: list[tuple[int, ...]],
    *,
    target_size: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Choose a seeded subset of whole orbits with exactly ``target_size`` states."""
    reachable: dict[int, tuple[int, ...]] = {0: tuple()}
    for raw_orbit_index in rng.permutation(len(orbits)):
        orbit_index = int(raw_orbit_index)
        orbit_size = len(orbits[orbit_index])
        for current_size in sorted(tuple(reachable), reverse=True):
            candidate_size = current_size + orbit_size
            if candidate_size > target_size or candidate_size in reachable:
                continue
            reachable[candidate_size] = (*reachable[current_size], orbit_index)
        if target_size in reachable:
            return reachable[target_size]
    raise ValueError(
        f"Cannot construct an orbit-complete holdout with {target_size} states "
        f"from orbit sizes {[len(orbit) for orbit in orbits]}."
    )


def make_d4_group_holdout_split(
    *,
    train_size: int = 450,
    test_size: int = 348,
    train_seed: int = 0,
    test_seed: int = 0,
) -> DatasetSplit:
    """Create a balanced split with no exact or D4-orbit train/test overlap.

    The test set is selected from complete D4 orbits using ``test_seed``. Training
    examples are sampled only from the remaining orbits using ``train_seed``.
    Keeping ``test_seed`` fixed therefore gives one holdout shared by every model,
    while equal training seeds produce nested training sets across sample sizes.
    """
    train_per_class = _check_balanced_size(train_size, "train_size")
    test_per_class = _check_balanced_size(test_size, "test_size")
    x, y, labels = generate_all_states()
    d4_orbits = make_d4_orbits(x=x)

    train_indices: list[int] = []
    test_indices: list[int] = []
    train_orbit_ids: set[int] = set()
    test_orbit_ids: set[int] = set()

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_orbit_items = [
            (orbit_id, orbit)
            for orbit_id, orbit in enumerate(d4_orbits)
            if str(labels[orbit[0]]) == class_name
        ]
        class_orbits = [orbit for _, orbit in class_orbit_items]
        holdout_rng = np.random.default_rng(
            np.random.SeedSequence([int(test_seed), 0xD4, class_index, 0])
        )
        selected_positions = set(
            _choose_orbits_with_exact_size(
                class_orbits,
                target_size=test_per_class,
                rng=holdout_rng,
            )
        )

        class_test_indices: list[int] = []
        class_train_pool: list[int] = []
        for position, (orbit_id, orbit) in enumerate(class_orbit_items):
            if position in selected_positions:
                class_test_indices.extend(orbit)
                test_orbit_ids.add(orbit_id)
            else:
                class_train_pool.extend(orbit)

        if len(class_train_pool) < train_per_class:
            raise ValueError(
                f"Leak-free balanced split is impossible for class {class_name!r}: "
                f"the orbit holdout leaves {len(class_train_pool)} states, but "
                f"train_size={train_size} requires {train_per_class}."
            )

        train_rng = np.random.default_rng(
            np.random.SeedSequence([int(train_seed), 0xD4, class_index, 1])
        )
        selected_train = train_rng.permutation(class_train_pool)[:train_per_class]
        selected_train_set = set(int(index) for index in selected_train)
        train_indices.extend(int(index) for index in selected_train)
        test_indices.extend(class_test_indices)
        train_orbit_ids.update(
            orbit_id
            for orbit_id, orbit in class_orbit_items
            if any(index in selected_train_set for index in orbit)
        )

    train_order_rng = np.random.default_rng(
        np.random.SeedSequence([int(train_seed), 0xD4, 99, 2])
    )
    test_order_rng = np.random.default_rng(
        np.random.SeedSequence([int(test_seed), 0xD4, 99, 3])
    )
    train_indices = train_order_rng.permutation(train_indices).astype(int).tolist()
    test_indices = test_order_rng.permutation(test_indices).astype(int).tolist()

    exact_overlap_count = len(set(train_indices) & set(test_indices))
    orbit_overlap_count = len(train_orbit_ids & test_orbit_ids)
    if exact_overlap_count or orbit_overlap_count:
        raise RuntimeError(
            "Internal split error: expected exact and D4-orbit disjointness, got "
            f"exact_overlap={exact_overlap_count}, orbit_overlap={orbit_overlap_count}."
        )

    train_digest = _index_set_digest(train_indices)
    test_digest = _index_set_digest(test_indices)
    metadata: dict[str, int | bool | str | float | tuple[int, ...]] = {
        "seed": train_seed,
        "train_seed": train_seed,
        "test_seed": test_seed,
        "train_size": train_size,
        "test_size": test_size,
        "requested_disjoint": True,
        "actual_disjoint": True,
        "overlap_count": 0,
        "split_mode": LEAK_FREE_SPLIT_PROTOCOL,
        "split_protocol": LEAK_FREE_SPLIT_PROTOCOL,
        "split_group": "D4",
        "group_disjoint": True,
        "orbit_overlap_count": 0,
        "train_orbit_count": len(train_orbit_ids),
        "test_orbit_count": len(test_orbit_ids),
        "train_index_digest": train_digest,
        "test_index_digest": test_digest,
        "split_index_digest": sha256(
            f"{train_digest}:{test_digest}".encode("ascii")
        ).hexdigest(),
        "requested_epsilon": 0.0,
        "actual_epsilon": 0.0,
        "corrupted_orbit_count": 0,
        "corruption_mode": "none",
        "broken_K": "D4",
        "corrupted_orbit_indices": tuple(),
    }

    return DatasetSplit(
        x_train=x[train_indices],
        y_train=y[train_indices],
        train_labels=labels[train_indices],
        x_test=x[test_indices],
        y_test=y[test_indices],
        test_labels=labels[test_indices],
        metadata=metadata,
    )


def _make_balanced_split_from_labels(
    *,
    x: np.ndarray,
    labels: np.ndarray,
    train_size: int = 450,
    test_size: int = 600,
    seed: int = 0,
    disjoint: bool = True,
    allow_overlap_if_needed: bool = True,
) -> tuple[list[int], list[int], bool, int]:
    train_per_class = _check_balanced_size(train_size, "train_size")
    test_per_class = _check_balanced_size(test_size, "test_size")
    rng = np.random.default_rng(seed)

    train_indices: list[int] = []
    test_indices: list[int] = []
    used_fallback = False

    for class_name in CLASS_NAMES:
        class_indices = np.flatnonzero(labels == class_name)
        if len(class_indices) < max(train_per_class, test_per_class):
            raise ValueError(
                f"Class {class_name!r} has {len(class_indices)} states, fewer than required "
                f"max({train_per_class}, {test_per_class})."
            )
        shuffled = rng.permutation(class_indices)
        can_be_disjoint = len(class_indices) >= train_per_class + test_per_class
        if disjoint and can_be_disjoint:
            train_indices.extend(shuffled[:train_per_class].tolist())
            test_indices.extend(
                shuffled[train_per_class : train_per_class + test_per_class].tolist()
            )
        elif disjoint and not can_be_disjoint and not allow_overlap_if_needed:
            raise ValueError(
                f"Cannot make disjoint balanced split for class {class_name!r}: "
                f"need {train_per_class + test_per_class}, have {len(class_indices)}."
            )
        else:
            used_fallback = True
            train_indices.extend(shuffled[:train_per_class].tolist())
            test_indices.extend(rng.choice(class_indices, size=test_per_class, replace=False).tolist())

    train_indices = rng.permutation(np.asarray(train_indices, dtype=int)).tolist()
    test_indices = rng.permutation(np.asarray(test_indices, dtype=int)).tolist()
    overlap_count = len(set(train_indices) & set(test_indices))
    return train_indices, test_indices, used_fallback, overlap_count


def make_balanced_split(
    *,
    train_size: int = 450,
    test_size: int = 600,
    seed: int = 0,
    disjoint: bool = True,
    allow_overlap_if_needed: bool = True,
) -> DatasetSplit:
    """Create balanced train/test splits."""
    x, y, labels = generate_all_states()
    train_indices, test_indices, used_fallback, overlap_count = _make_balanced_split_from_labels(
        x=x,
        labels=labels,
        train_size=train_size,
        test_size=test_size,
        seed=seed,
        disjoint=disjoint,
        allow_overlap_if_needed=allow_overlap_if_needed,
    )

    metadata: dict[str, int | bool | str | float | tuple[int, ...]] = {
        "seed": seed,
        "train_size": train_size,
        "test_size": test_size,
        "requested_disjoint": disjoint,
        "actual_disjoint": overlap_count == 0,
        "overlap_count": overlap_count,
        "split_mode": "independent_fallback" if used_fallback else "disjoint",
        "requested_epsilon": 0.0,
        "actual_epsilon": 0.0,
        "corrupted_orbit_count": 0,
        "corruption_mode": "none",
        "broken_K": "D4",
        "corrupted_orbit_indices": tuple(),
    }

    return DatasetSplit(
        x_train=x[train_indices],
        y_train=y[train_indices],
        train_labels=labels[train_indices],
        x_test=x[test_indices],
        y_test=y[test_indices],
        test_labels=labels[test_indices],
        metadata=metadata,
    )


def make_approximate_symmetry_split(
    *,
    train_size: int = 450,
    test_size: int = 600,
    seed: int = 0,
    disjoint: bool = True,
    allow_overlap_if_needed: bool = True,
    epsilon: float = 0.0,
    corruption_mode: str = "swap_circle_cross",
    broken_K: str = "C4",
) -> DatasetSplit:
    """Create balanced train/test splits with epsilon-corrupted K-orbit labels."""
    x, y, labels = generate_all_states()
    corrupted_labels, corruption_meta = make_corrupted_labels(
        x=x,
        labels=labels,
        epsilon=epsilon,
        seed=seed,
        corruption_mode=corruption_mode,
        broken_K=broken_K,
    )
    train_indices, test_indices, used_fallback, overlap_count = _make_balanced_split_from_labels(
        x=x,
        labels=corrupted_labels,
        train_size=train_size,
        test_size=test_size,
        seed=seed,
        disjoint=disjoint,
        allow_overlap_if_needed=allow_overlap_if_needed,
    )

    metadata: dict[str, int | bool | str | float | tuple[int, ...]] = {
        "seed": seed,
        "train_size": train_size,
        "test_size": test_size,
        "requested_disjoint": disjoint,
        "actual_disjoint": overlap_count == 0,
        "overlap_count": overlap_count,
        "split_mode": "independent_fallback" if used_fallback else "disjoint",
        "requested_epsilon": 0.0,
        "actual_epsilon": 0.0,
        "corrupted_orbit_count": 0,
        "corruption_mode": "none",
        "broken_K": broken_K,
        "corrupted_orbit_indices": tuple(),
    }
    metadata.update(corruption_meta)
    return DatasetSplit(
        x_train=x[train_indices],
        y_train=y[train_indices],
        train_labels=corrupted_labels[train_indices],
        x_test=x[test_indices],
        y_test=y[test_indices],
        test_labels=corrupted_labels[test_indices],
        metadata=metadata,
    )


def labels_are_d4_invariant() -> bool:
    x, _, labels = generate_all_states()
    label_by_board = {tuple(board.astype(int).tolist()): label for board, label in zip(x, labels)}
    for board, label in zip(x, labels):
        for perm in PERMUTATIONS.values():
            transformed = tuple(apply_permutation_to_board(board.astype(int), perm).tolist())
            if label_by_board.get(transformed) != label:
                return False
    return True
