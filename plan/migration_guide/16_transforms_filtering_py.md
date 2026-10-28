# Migration Map: `src/preprocessing_core/transforms/filtering.py`

## Purpose

Generic post-split row filters.

These transforms may change split lengths, so they must use `select_time_rows` to preserve alignment.

## Scope

Owns:

- public generic timestep filtering
- quantile-based row filtering

Does not own:

- domain-specific quality filters
- source-specific timestamp trimming
- channel quality selection based on external references

## `filter_quantile_timesteps` pseudocode

```python
"""Generic row-filtering transforms."""

import numpy as np

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.selection import select_time_rows
from preprocessing_core.validation import ConfigError, PipelineError


def filter_quantile_timesteps(
    dataset: BaseDataset,
    *,
    state: dict[str, object],
    fit: bool,
    low_quantile: float = 1.0,
    high_quantile: float = 99.0,
    min_valid_fraction: float = 1.0,
    require_positive: bool = False,
) -> BaseDataset:
    """Drop time rows with too many values outside configured quantile bounds."""
    _validate_quantile_params(low_quantile, high_quantile, min_valid_fraction)

    data = dataset.data
    finite = np.isfinite(data)
    valid = finite.copy()
    if require_positive:
        valid &= data > 0

    # Compute bounds from this split only. If train-fitted bounds are needed,
    # implement a separate fitted transform with explicit state.
    low = np.nanpercentile(data, low_quantile, axis=0)
    high = np.nanpercentile(data, high_quantile, axis=0)
    in_range = valid & (data >= low) & (data <= high)

    fraction = in_range.reshape(data.shape[0], -1).mean(axis=1)
    keep = fraction >= min_valid_fraction

    if not np.any(keep):
        raise PipelineError("filter_quantile_timesteps would remove all rows")

    return select_time_rows(dataset, keep)
```

## Fit/state policy

This transform should not silently use `state`. If a future version needs train-fitted quantile bounds:

- create a separate transform, e.g. `fitted_quantile_filter`
- use `fit: train`
- store serializable bounds in `state`
- test val/test do not fit themselves

## Error handling

- Invalid quantiles: `ConfigError`
- All rows removed: `PipelineError`
- Missing/invalid data shape should already fail dataset validation

## Logging

No logging here. Pipeline logs before/after transform and shape.

## Human readability

- Keep filter criteria explicit.
- Avoid combining many unrelated filters into one transform.
- If domain logic appears, move it to a private plugin transform.

## Tests

- row filtering slices timestamps/labels/channel labels/aligned metadata
- all rows removed fails
- invalid quantile params fail
- transform preserves orchestrator-owned fields
