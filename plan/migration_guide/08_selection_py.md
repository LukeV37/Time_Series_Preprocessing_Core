# Migration Map: `src/preprocessing_core/selection.py`

## Purpose

Provide the only framework-approved helpers for row and channel selection.

These helpers are critical for safe time-series preprocessing because they preserve alignment between data, timestamps, labels, channel labels, names, and aligned metadata.

## Scope

Owns:

- `select_time_rows(dataset, keep_or_indices)`
- `select_channels(dataset, keep_or_indices)`
- optional small helpers to normalize masks/indices

Does not own:

- split policy computation
- transform decisions about which rows/channels to keep
- validation beyond immediate selection sanity checks

## `select_time_rows` pseudocode

```python
"""Alignment-preserving dataset selection helpers."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.validation import PipelineError


def select_time_rows(dataset: BaseDataset, keep_or_indices) -> BaseDataset:
    """Return a dataset containing selected time rows with all time-aligned fields sliced."""
    selector = _normalize_row_selector(keep_or_indices, n_rows=len(dataset))

    metadata = copy.deepcopy(dataset.metadata)
    aligned_time = metadata.get("aligned_time")
    if aligned_time is not None:
        if not isinstance(aligned_time, dict):
            raise PipelineError("metadata['aligned_time'] must be a mapping")
        metadata["aligned_time"] = {
            key: np.asarray(value)[selector].copy()
            for key, value in aligned_time.items()
        }

    result = BaseDataset(
        data=np.asarray(dataset.data)[selector].copy(),
        timestamps=None if dataset.timestamps is None else np.asarray(dataset.timestamps)[selector].copy(),
        labels=None if dataset.labels is None else np.asarray(dataset.labels)[selector].copy(),
        channel_labels=None if dataset.channel_labels is None else np.asarray(dataset.channel_labels)[selector].copy(),
        channel_names=None if dataset.channel_names is None else list(dataset.channel_names),
        feature_names=None if dataset.feature_names is None else list(dataset.feature_names),
        metadata=metadata,
        split=dataset.split,
        data_tag=dataset.data_tag,
        preprocess_steps=list(dataset.preprocess_steps),
    )
    result.validate()
    return result
```

## `select_channels` pseudocode

```python
def select_channels(dataset: BaseDataset, keep_or_indices) -> BaseDataset:
    """Return a dataset with selected channels and channel-aligned fields updated."""
    selector = _normalize_channel_selector(keep_or_indices, n_channels=dataset.n_channels)

    result = dataset.copy()
    result.data = np.asarray(dataset.data)[:, selector, :].copy()
    if dataset.channel_labels is not None:
        result.channel_labels = np.asarray(dataset.channel_labels)[:, selector].copy()
    if dataset.channel_names is not None:
        result.channel_names = _select_list(dataset.channel_names, selector)

    # If public convention later supports metadata['aligned_channel'], slice it here.
    # V1 only guarantees data/channel_labels/channel_names.

    result.validate()
    return result
```

## Selector normalization

```python
def _normalize_row_selector(keep_or_indices, *, n_rows: int):
    selector = np.asarray(keep_or_indices)
    if selector.dtype == bool:
        if selector.shape != (n_rows,):
            raise PipelineError(f"Boolean row selector length {selector.size} does not match T={n_rows}")
        return selector
    if not np.issubdtype(selector.dtype, np.integer):
        raise PipelineError("Row selector must be boolean mask or integer indices")
    if selector.ndim != 1:
        raise PipelineError("Row index selector must be 1D")
    if np.any(selector < 0) or np.any(selector >= n_rows):
        raise PipelineError("Row selector contains out-of-range indices")
    return selector.astype(int, copy=False)
```

Do equivalent validation for channels.

## Copying policy

- V1 should copy selected arrays to avoid accidental aliasing across splits.
- Do not optimize with views unless there is a measured memory problem and tests are added.

## Human readability

- Keep all alignment slicing here.
- Transforms should not manually slice `timestamps`, `labels`, or `metadata['aligned_time']`.
- If a future aligned axis is added, add it here and test it here.

## Error handling

- Use `PipelineError` for invalid selectors or impossible alignment state.
- Include whether the failure is row or channel selection.

## Logging

No logging here. Pipeline logs transform-level context.

## Tests

- row selection slices data, timestamps, labels, channel labels, and aligned metadata
- channel selection slices data, channel labels, and channel names
- boolean and integer selectors both work
- invalid selector shapes fail
- split outputs do not alias full dataset arrays
