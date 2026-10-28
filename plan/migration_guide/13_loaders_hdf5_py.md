# Migration Map: `src/preprocessing_core/loaders/hdf5.py`

## Purpose

Provide a minimal public HDF5 loader using optional `h5py`.

This is not a general HDF5 framework. It reads named datasets into memory.

## Scope

Owns:

- lazy import of `h5py`
- reading named datasets by key
- shape normalization
- returning `BaseDataset`

Does not own:

- streaming/chunked loading
- complex group discovery
- private experiment formats

## Pseudocode

```python
"""Minimal HDF5 loader for named array datasets."""

import numpy as np

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.loaders.utils import normalize_data_shape
from preprocessing_core.validation import PipelineError


def load_hdf5(
    *,
    path: str,
    data_key: str = "data",
    timestamps_key: str | None = None,
    labels_key: str | None = None,
    channel_labels_key: str | None = None,
) -> BaseDataset:
    """Load named datasets from an HDF5 file into memory."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("The hdf5 loader requires optional dependency 'h5py'") from exc

    with h5py.File(path, "r") as handle:
        if data_key not in handle:
            raise PipelineError(f"HDF5 file {path!r} is missing data_key {data_key!r}")
        data = normalize_data_shape(handle[data_key][...])
        timestamps = _read_optional_dataset(handle, timestamps_key)
        labels = _read_optional_dataset(handle, labels_key)
        channel_labels = _read_optional_dataset(handle, channel_labels_key)

    dataset = BaseDataset(
        data=data,
        timestamps=timestamps,
        labels=labels,
        channel_labels=channel_labels,
        metadata={"source_format": "hdf5"},
    )
    dataset.validate()
    return dataset
```

## Error handling

- Missing optional dependency: `ImportError` with install hint.
- Missing configured key: `PipelineError` with file/key.
- Invalid shape: validation error.

## Logging

No logging.

## Tests

- skip if `h5py` unavailable
- minimal data key works
- optional arrays work
- missing key fails clearly
