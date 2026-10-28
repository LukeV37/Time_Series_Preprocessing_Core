# Migration Map: `src/preprocessing_core/loaders/npy.py`

## Purpose

Load a `.npy` array into `BaseDataset` with canonical `(T, C, F)` shape.

## Scope

Owns:

- `np.load` for `.npy`
- optional sidecar arrays passed by path
- shape normalization
- returning valid `BaseDataset`

Does not own:

- splitting
- preprocessing transforms
- fitting/scaling

## Pseudocode

```python
"""Generic NumPy `.npy` loader."""

from pathlib import Path
import numpy as np

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.loaders.utils import normalize_data_shape, load_optional_array


def load_npy(
    *,
    path: str,
    timestamps_path: str | None = None,
    labels_path: str | None = None,
    channel_labels_path: str | None = None,
    channel_names: list[str] | None = None,
    feature_names: list[str] | None = None,
) -> BaseDataset:
    """Load a `.npy` time-series array into canonical dataset form."""
    data = normalize_data_shape(np.load(Path(path), allow_pickle=False))
    dataset = BaseDataset(
        data=data,
        timestamps=load_optional_array(timestamps_path),
        labels=load_optional_array(labels_path),
        channel_labels=load_optional_array(channel_labels_path),
        channel_names=channel_names,
        feature_names=feature_names,
        metadata={"source_format": "npy"},
    )
    dataset.validate()
    return dataset
```

## Helper note

The plan did not list `loaders/utils.py`, but a small private helper module is acceptable if it avoids duplicating shape normalization across loaders. Keep it private and focused.

## Error handling

- Let `FileNotFoundError` surface for missing file paths.
- Use `PipelineError` or `ValueError` for unsupported dimensions in `normalize_data_shape`.
- Do not catch broadly.

## Logging

No logging in loader. Pipeline logs selected loader.

## Tests

- 1D becomes `(T,1,1)`
- 2D becomes `(T,C,1)`
- 3D unchanged
- timestamps/labels validate
- wrong sidecar length fails
