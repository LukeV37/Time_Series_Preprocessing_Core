# Migration Map: `src/preprocessing_core/splits/ranges.py`

## Purpose

Implement explicit half-open index range split policy.

This is useful for reproducible externally decided boundaries and simple testing.

## Scope

Owns:

- parsing train/val/test `[start, stop)` ranges
- optional gap trimming
- basic bounds checks

Does not own:

- row selection
- timestamp-based splitting
- group/year/campaign semantics

## Pseudocode

```python
"""Explicit index range split policy."""

import numpy as np

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.validation import ConfigError


def index_ranges_split(
    dataset: BaseDataset,
    *,
    train: tuple[int, int] | list[int],
    val: tuple[int, int] | list[int],
    test: tuple[int, int] | list[int],
    gap: int = 0,
) -> dict[str, np.ndarray]:
    """Return train/val/test indices from explicit half-open ranges."""
    n = len(dataset)
    ranges = {
        "train": _validate_range("train", train, n),
        "val": _validate_range("val", val, n),
        "test": _validate_range("test", test, n),
    }

    result = {
        name: np.arange(start, stop)
        for name, (start, stop) in ranges.items()
    }

    if gap:
        _validate_gap(gap)
        # For adjacent chronological ranges, trim both sides of boundaries.
        result["train"] = result["train"][: max(0, len(result["train"]) - gap)]
        result["val"] = result["val"][gap: max(gap, len(result["val"]) - gap)]
        result["test"] = result["test"][gap:]

    _reject_empty(result)
    return result
```

## Validation

This policy should reject:

- malformed range not length 2
- negative start/stop
- start >= stop
- stop > `len(dataset)`
- overlap between any split ranges
- empty split after gap

Coverage of all rows is **not required**. Explicit ranges may intentionally leave rows unused.

## Error handling

Use `ConfigError` for invalid range config.

## Logging

No logging.

## Human readability

- Keep semantics half-open and documented.
- Do not add timestamp parsing here.
- If users need timestamp ranges, add a future policy with docs/tests.

## Tests

- simple ranges work
- bounds fail
- overlap fails
- gaps trim expected rows
- empty split fails
