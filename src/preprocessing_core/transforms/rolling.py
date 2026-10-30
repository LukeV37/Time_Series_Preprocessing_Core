"""Rolling-window repair transforms."""

from __future__ import annotations

import numpy as np

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.validation import ConfigError


def repair_bollinger_outliers(
    dataset: BaseDataset,
    *,
    state: dict[str, object],
    fit: bool,
    window: int,
    n_bands: float = 3.0,
    min_periods: int | None = None,
) -> BaseDataset:
    """Replace trailing-window Bollinger outliers with the rolling mean.

    The window uses only previously repaired values, so the current value cannot
    weaken its own detection band and no future rows are used.
    """
    _ = state, fit
    window, min_periods = _validate_params(window=window, n_bands=n_bands, min_periods=min_periods)

    result = dataset.copy()
    data = np.asarray(result.data, dtype=float).copy()

    for channel in range(data.shape[1]):
        for feature in range(data.shape[2]):
            data[:, channel, feature] = _repair_series(
                data[:, channel, feature],
                window=window,
                n_bands=float(n_bands),
                min_periods=min_periods,
            )

    result.data = data
    return result


def _validate_params(
    *,
    window: int,
    n_bands: float,
    min_periods: int | None,
) -> tuple[int, int]:
    """Validate rolling transform parameters."""
    if not isinstance(window, int) or window <= 0:
        raise ConfigError("repair_bollinger_outliers window must be a positive integer")
    if float(n_bands) <= 0:
        raise ConfigError("repair_bollinger_outliers n_bands must be positive")

    if min_periods is None:
        min_periods = window
    if not isinstance(min_periods, int) or min_periods <= 0:
        raise ConfigError("repair_bollinger_outliers min_periods must be a positive integer")
    if min_periods > window:
        raise ConfigError("repair_bollinger_outliers min_periods cannot exceed window")

    return window, min_periods


def _repair_series(
    values: np.ndarray,
    *,
    window: int,
    n_bands: float,
    min_periods: int,
) -> np.ndarray:
    """Return one repaired one-dimensional series."""
    repaired = np.asarray(values, dtype=float).copy()

    for index in range(repaired.size):
        start = max(0, index - window)
        history = repaired[start:index]
        if history.size < min_periods:
            continue

        mean = float(np.mean(history))
        std = float(np.std(history))
        if std == 0.0:
            continue

        if abs(float(repaired[index]) - mean) > n_bands * std:
            repaired[index] = mean

    return repaired
