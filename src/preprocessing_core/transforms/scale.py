"""Scaling transforms."""

from __future__ import annotations

import numpy as np

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.validation import PipelineError


def standard_scale(
    dataset: BaseDataset,
    *,
    state: dict[str, object],
    fit: bool,
    epsilon: float = 1.0e-8,
) -> BaseDataset:
    """Scale each channel-feature series using train-fitted mean and std.

    The fitted state contains one mean and one std for every `(channel, feature)`
    pair, both with shape `(C, F)`.
    """
    if fit:
        mean = np.mean(dataset.data, axis=0)
        std = np.std(dataset.data, axis=0)
        state["mean"] = mean
        state["std"] = std

    if "mean" not in state or "std" not in state:
        raise PipelineError("standard_scale requires fitted 'mean' and 'std' state")

    mean = np.asarray(state["mean"])
    std = np.asarray(state["std"])
    if mean.shape != dataset.data.shape[1:] or std.shape != dataset.data.shape[1:]:
        raise PipelineError(
            "standard_scale fitted state shape does not match dataset channel-feature shape"
        )

    safe_std = np.where(std <= float(epsilon), 1.0, std)
    result = dataset.copy()
    result.data = (np.asarray(result.data) - mean[None, :, :]) / safe_std[None, :, :]
    return result
