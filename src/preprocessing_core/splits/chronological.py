"""Chronological split policy for time-series datasets."""

from __future__ import annotations

import numpy as np

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.validation import ConfigError


def chronological_split(
    dataset: BaseDataset,
    *,
    ratios: list[float] | tuple[float, float, float],
    gap: int = 0,
) -> dict[str, np.ndarray]:
    """Return contiguous train/val/test row indices in time order.

    Train and validation lengths use floor rounding. Test receives the
    remaining rows. If `gap` is positive, rows near split boundaries are dropped
    to reduce leakage:

    - `gap` rows from the end of train
    - `gap` rows from the start and end of validation
    - `gap` rows from the start of test
    """
    train_ratio, val_ratio, _ = _validate_ratios(ratios)
    _validate_gap(gap)

    n_rows = len(dataset)
    train_end = int(np.floor(n_rows * train_ratio))
    val_end = train_end + int(np.floor(n_rows * val_ratio))

    splits = {
        "train": np.arange(0, train_end),
        "val": np.arange(train_end, val_end),
        "test": np.arange(val_end, n_rows),
    }

    if gap:
        splits = {
            "train": splits["train"][: max(0, len(splits["train"]) - gap)],
            "val": splits["val"][gap : max(gap, len(splits["val"]) - gap)],
            "test": splits["test"][gap:],
        }

    _reject_empty_splits(splits, n_rows=n_rows, gap=gap, ratios=ratios)
    return splits


def _validate_ratios(ratios: list[float] | tuple[float, float, float]) -> tuple[float, float, float]:
    """Return validated train/val/test ratios."""
    if len(ratios) != 3:
        raise ConfigError("chronological split ratios must contain exactly three values")

    train_ratio, val_ratio, test_ratio = (float(value) for value in ratios)
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ConfigError("chronological split ratios must all be positive")

    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ConfigError(f"chronological split ratios must sum to 1.0; got {total}")

    return train_ratio, val_ratio, test_ratio


def _validate_gap(gap: int) -> None:
    """Raise if gap is not a non-negative integer."""
    if not isinstance(gap, int):
        raise ConfigError("chronological split gap must be an integer")
    if gap < 0:
        raise ConfigError("chronological split gap must be non-negative")


def _reject_empty_splits(
    splits: dict[str, np.ndarray],
    *,
    n_rows: int,
    gap: int,
    ratios: list[float] | tuple[float, float, float],
) -> None:
    """Raise if rounding or gap removal leaves any split empty."""
    empty = [name for name, indices in splits.items() if len(indices) == 0]
    if empty:
        raise ConfigError(
            "chronological split produced empty splits "
            f"{empty} for n_rows={n_rows}, ratios={list(ratios)}, gap={gap}"
        )
