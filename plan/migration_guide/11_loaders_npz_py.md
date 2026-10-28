# Migration Map: `src/preprocessing_core/loaders/npz.py`

## Purpose

Load a `.npz` archive containing named arrays into `BaseDataset`.

This should be the first implemented loader because it is ideal for synthetic tests.

## Scope

Owns:

- opening `.npz`
- reading configured keys
- shape normalization
- optional metadata fields
- returning valid `BaseDataset`

## Pseudocode

```python
"""Generic NumPy `.npz` loader."""

from pathlib import Path
import numpy as np

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.loaders.utils import normalize_data_shape
from preprocessing_core.validation import PipelineError


def load_npz(
    *,
    path: str,
    data_key: str = "data",
    timestamps_key: str | None = None,
    labels_key: str | None = None,
    channel_labels_key: str | None = None,
    channel_names_key: str | None = None,
    feature_names_key: str | None = None,
) -> BaseDataset:
    """Load named arrays from a `.npz` archive."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if data_key not in archive:
            raise PipelineError(f"NPZ file {path!r} is missing data_key {data_key!r}")

        data = normalize_data_shape(archive[data_key])
        dataset = BaseDataset(
            data=data,
            timestamps=_optional_array(archive, timestamps_key),
            labels=_optional_array(archive, labels_key),
            channel_labels=_optional_array(archive, channel_labels_key),
            channel_names=_optional_names(archive, channel_names_key),
            feature_names=_optional_names(archive, feature_names_key),
            metadata={"source_format": "npz"},
        )

    dataset.validate()
    return dataset
```

## Name conversion

`channel_names` and `feature_names` should be converted to `list[str]`. Reject object arrays if `allow_pickle=False` cannot load them safely. Prefer string arrays.

## Error handling

- Missing `data_key`: `PipelineError` with file/key.
- Missing optional key when configured: either raise clearly or document that absent optional means `None`. Prefer raise if user explicitly configured it.
- Invalid shapes fail through `dataset.validate()`.

## Logging

No logging.

## Tests

- minimal `data` only works
- optional timestamps/labels work
- missing configured key fails clearly
- wrong shape fails
- loader called once in end-to-end pipeline test
