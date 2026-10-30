"""Alignment-preserving dataset selection helpers."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.validation import PipelineError


def select_time_rows(dataset: BaseDataset, keep_or_indices: Any) -> BaseDataset:
    """Return a dataset containing selected time rows with aligned fields sliced."""
    selector = _normalize_selector(keep_or_indices, size=len(dataset), axis_name="row")

    metadata = copy.deepcopy(dataset.metadata)
    aligned_time = metadata.get("aligned_time")
    if aligned_time is not None:
        if not isinstance(aligned_time, dict):
            raise PipelineError("metadata['aligned_time'] must be a mapping")
        metadata["aligned_time"] = {
            key: np.asarray(value)[selector].copy() for key, value in aligned_time.items()
        }

    result = BaseDataset(
        data=np.asarray(dataset.data)[selector].copy(),
        timestamps=None
        if dataset.timestamps is None
        else np.asarray(dataset.timestamps)[selector].copy(),
        labels=None if dataset.labels is None else np.asarray(dataset.labels)[selector].copy(),
        channel_labels=None
        if dataset.channel_labels is None
        else np.asarray(dataset.channel_labels)[selector].copy(),
        channel_names=None if dataset.channel_names is None else list(dataset.channel_names),
        feature_names=None if dataset.feature_names is None else list(dataset.feature_names),
        metadata=metadata,
        split=dataset.split,
        data_tag=dataset.data_tag,
        preprocess_steps=list(dataset.preprocess_steps),
    )
    result.validate()
    return result


def select_channels(dataset: BaseDataset, keep_or_indices: Any) -> BaseDataset:
    """Return a dataset with selected channels and channel-aligned fields updated."""
    selector = _normalize_selector(keep_or_indices, size=dataset.n_channels, axis_name="channel")

    result = dataset.copy()
    result.data = np.asarray(dataset.data)[:, selector, :].copy()
    if dataset.channel_labels is not None:
        result.channel_labels = np.asarray(dataset.channel_labels)[:, selector].copy()
    if dataset.channel_names is not None:
        result.channel_names = _select_list(dataset.channel_names, selector)
    result.validate()
    return result


def _normalize_selector(keep_or_indices: Any, *, size: int, axis_name: str) -> np.ndarray:
    selector = np.asarray(keep_or_indices)
    if selector.dtype == bool:
        if selector.shape != (size,):
            raise PipelineError(
                f"Boolean {axis_name} selector length {selector.size} does not match {size}"
            )
        return selector
    if selector.ndim != 1:
        raise PipelineError(f"{axis_name.title()} index selector must be 1D")
    if not np.issubdtype(selector.dtype, np.integer):
        raise PipelineError(f"{axis_name.title()} selector must be boolean mask or integer indices")
    if np.any(selector < 0) or np.any(selector >= size):
        raise PipelineError(f"{axis_name.title()} selector contains out-of-range indices")
    return selector.astype(int, copy=False)


def _select_list(values: list[str], selector: np.ndarray) -> list[str]:
    if selector.dtype == bool:
        return [value for value, keep in zip(values, selector, strict=True) if keep]
    return [values[int(index)] for index in selector]
