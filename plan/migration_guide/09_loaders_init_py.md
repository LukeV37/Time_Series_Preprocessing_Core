# Migration Map: `src/preprocessing_core/loaders/__init__.py`

## Purpose

Expose built-in loader functions and optionally provide a function to register them.

## Scope

Owns:

- imports for built-in generic loaders
- optional `register_builtin_loaders()` helper

Does not own:

- loader implementation
- optional dependency imports that break base package import

## Pseudocode

```python
"""Built-in generic loaders for preprocessing_core."""

from preprocessing_core.loaders.npy import load_npy
from preprocessing_core.loaders.npz import load_npz
from preprocessing_core.loaders.csv import load_csv_timeseries

# Do not import hdf5 here if h5py is optional and imported at module import time.


def register_builtin_loaders(*, replace: bool = False) -> None:
    from preprocessing_core.registry import register_loader

    register_loader("npy", load_npy, replace=replace)
    register_loader("npz", load_npz, replace=replace)
    register_loader("csv", load_csv_timeseries, replace=replace)

    # Import hdf5 lazily so users without h5py can still import preprocessing_core.
    try:
        from preprocessing_core.loaders.hdf5 import load_hdf5
    except ImportError:
        load_hdf5 = None
    if load_hdf5 is not None:
        register_loader("hdf5", load_hdf5, replace=replace)
```

## Human readability

- Keep optional dependency behavior obvious.
- Do not silently register unavailable optional loaders unless docs explain it.

## Error handling

If `hdf5` is requested but `h5py` is missing, the `hdf5` loader should raise `ImportError` with an installation hint.

## Logging

No logging.

## Tests

- importing `preprocessing_core.loaders` works without `h5py`
- built-in loaders can be registered
- missing optional loader behavior is clear
