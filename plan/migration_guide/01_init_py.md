# Migration Map: `src/preprocessing_core/__init__.py`

## Purpose

Package-level public exports. This file should make the public API easy to discover without hiding where behavior lives.

## Scope

Export only stable public symbols:

- `load_dataset_bundle`
- `build_bundle`
- `load_config`
- `BaseDataset`
- `DatasetBundle`
- `register_loader`
- `register_transform`
- `register_split_policy`
- exception types, if implemented publicly

Do not put logic here.

## Pseudocode

```python
"""Public API for the preprocessing core mini-framework."""

from preprocessing_core.api import build_bundle, load_dataset_bundle
from preprocessing_core.config import load_config
from preprocessing_core.datasets import BaseDataset, DatasetBundle
from preprocessing_core.registry import (
    register_loader,
    register_split_policy,
    register_transform,
)
from preprocessing_core.validation import ConfigError, PipelineError

__all__ = [
    "BaseDataset",
    "DatasetBundle",
    "ConfigError",
    "PipelineError",
    "build_bundle",
    "load_config",
    "load_dataset_bundle",
    "register_loader",
    "register_split_policy",
    "register_transform",
]
```

## Human readability rules

- Keep imports grouped and explicit.
- Do not re-export internal helpers like `ScheduledStep` unless there is a real public need.
- Do not perform built-in registration here if it causes import cycles. Prefer built-in registration inside `registry.py` or package init functions.

## Error handling

No error handling here.

## Logging

No logging here.

## Tests

- Import package and verify expected public names exist.
- Verify importing `preprocessing_core` has no side effects requiring optional dependencies like `h5py`.
