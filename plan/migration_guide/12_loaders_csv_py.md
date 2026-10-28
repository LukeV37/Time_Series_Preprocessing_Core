# Migration Map: `src/preprocessing_core/loaders/csv.py`

## Purpose

Load simple public CSV time-series data into `BaseDataset`.

CSV support should be intentionally modest. Complex CSV layouts belong in custom loaders.

## Scope

Owns:

- pandas CSV read
- timestamp column extraction
- optional label column extraction
- value column selection
- conversion to `(T, C, F)`

Does not own:

- per-channel labels from complex CSV layouts
- domain-specific parsing
- multi-file stitching

## Pseudocode

```python
"""Simple CSV time-series loader."""

import pandas as pd
import numpy as np

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.loaders.utils import normalize_data_shape
from preprocessing_core.validation import ConfigError


def load_csv_timeseries(
    *,
    path: str,
    value_columns: list[str] | None = None,
    timestamp_column: str | None = None,
    label_column: str | None = None,
) -> BaseDataset:
    """Load a simple tabular time series from CSV."""
    frame = pd.read_csv(path)

    timestamps = None
    if timestamp_column is not None:
        _require_columns(frame, [timestamp_column])
        timestamps = frame[timestamp_column].to_numpy()

    labels = None
    if label_column is not None:
        _require_columns(frame, [label_column])
        labels = frame[label_column].to_numpy()

    if value_columns is None:
        excluded = {timestamp_column, label_column, None}
        value_columns = [col for col in frame.columns if col not in excluded]
    _require_columns(frame, value_columns)

    values = frame[value_columns].to_numpy(dtype=float)
    data = normalize_data_shape(values)  # `(T, C)` -> `(T, C, 1)`

    dataset = BaseDataset(
        data=data,
        timestamps=timestamps,
        labels=labels,
        channel_names=list(value_columns),
        feature_names=["value"],
        metadata={"source_format": "csv"},
    )
    dataset.validate()
    return dataset
```

## Error handling

- Missing configured column: `ConfigError` or `PipelineError` with column name.
- Non-numeric value columns: let pandas/NumPy error surface, or wrap with column context.
- Do not infer complex layouts silently.

## Logging

No logging.

## Human readability

- Keep CSV rules simple and documented.
- If users need multi-index or per-channel labels, tell them to write a custom loader.

## Tests

- timestamp + labels + values works
- no timestamp works
- default value column inference works
- missing configured column fails
- non-numeric values fail clearly
