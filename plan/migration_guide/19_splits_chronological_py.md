# Migration Map: `src/preprocessing_core/splits/chronological.py`

## Purpose

Implement the default chronological time-series split policy.

This policy returns row indices only. It does not slice datasets.

## Scope

Owns:

- ratio validation
- contiguous index calculation
- optional leakage gap removal
- deterministic rounding rule

Does not own:

- row selection
- dataset copying
- bundle metadata creation, except optional dropped-row info if returned by convention later

## Pseudocode

```python
"""Chronological split policy."""

import numpy as np

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.validation import ConfigError


def chronological_split(
    dataset: BaseDataset,
    *,
    ratios: tuple[float, float, float] | list[float],
    gap: int = 0,
) -> dict[str, np.ndarray]:
    """Return contiguous chronological train/val/test row indices."""
    n = len(dataset)
    train_ratio, val_ratio, test_ratio = _validate_ratios(ratios)
    _validate_gap(gap)

    train_end = int(np.floor(n * train_ratio))
    val_end = train_end + int(np.floor(n * val_ratio))

    train = np.arange(0, train_end)
    val = np.arange(train_end, val_end)
    test = np.arange(val_end, n)

    if gap:
        train = train[: max(0, len(train) - gap)]
        val = val[gap: max(gap, len(val) - gap)]
        test = test[gap:]

    result = {"train": train, "val": val, "test": test}
    _reject_empty(result, n=n, gap=gap, ratios=ratios)
    return result
```

## Rounding rule

Use this exact rule and test it:

- train length: `floor(n * train_ratio)`
- val length: `floor(n * val_ratio)`
- test receives remainder

## Gap rule

For `train | val | test`:

- remove `gap` rows from end of train
- remove `gap` rows from start and end of val
- remove `gap` rows from start of test

Rows removed by gap belong to no split.

## Error handling

Use `ConfigError` for:

- ratios not length 3
- ratios not positive
- ratios not summing to 1 within tolerance
- negative gap
- empty split after rounding/gap

## Logging

No logging. Pipeline logs split sizes.

## Human readability

- Keep calculation explicit.
- Do not add random/grouped/year behavior here.
- If grouped policies are needed, implement private plugin policies.

## Tests

- deterministic boundaries
- remainder goes to test
- gap removes expected rows
- no overlap
- empty split fails
- invalid ratios fail
