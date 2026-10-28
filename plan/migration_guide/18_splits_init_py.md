# Migration Map: `src/preprocessing_core/splits/__init__.py`

## Purpose

Expose and register built-in generic split policies.

## Scope

Owns:

- imports for built-in split policies
- optional `register_builtin_split_policies()` helper

Does not own:

- split implementation
- split validation

## Pseudocode

```python
"""Built-in split policies for preprocessing_core."""

from preprocessing_core.splits.chronological import chronological_split
from preprocessing_core.splits.ranges import index_ranges_split
from preprocessing_core.splits.seasonal import seasonal_block_split


def register_builtin_split_policies(*, replace: bool = False) -> None:
    from preprocessing_core.registry import register_split_policy

    register_split_policy("chronological", chronological_split, replace=replace)
    register_split_policy("index_ranges", index_ranges_split, replace=replace)
    register_split_policy("seasonal_block", seasonal_block_split, replace=replace)
```

## Human readability

- Keep YAML names and function names clear.
- v1 built-ins are exactly `chronological`, `index_ranges`, and `seasonal_block`.
- Do not add grouped/year/campaign policies beyond those in v1.

## Tests

- built-ins register under expected names
