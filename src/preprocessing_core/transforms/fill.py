"""Missing-value fill transforms."""

from __future__ import annotations

import numpy as np

from preprocessing_core.datasets import BaseDataset


def fill_nan(
    dataset: BaseDataset,
    *,
    state: dict[str, object],
    fit: bool,
    value: float = 0.0,
) -> BaseDataset:
    """Replace NaN values in `dataset.data` with a constant value."""
    _ = state, fit

    result = dataset.copy()
    result.data = np.nan_to_num(np.asarray(result.data), nan=float(value), copy=True)
    return result
