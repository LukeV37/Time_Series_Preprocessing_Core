# Migration Map: `src/preprocessing_core/registry.py`

## Purpose

Provide explicit, in-process registration for loaders, transforms, and split policies.

This is a mini-framework extension point. It replaces dotted dynamic imports and avoids hidden auto-discovery.

## Scope

Owns:

- three dictionaries
- registration functions
- lookup functions
- duplicate-name and unknown-name errors
- built-in registration, if simple and non-cyclic

Does not own:

- config parsing
- callable execution
- transform scheduling

## Type aliases

```python
"""Explicit callable registries for preprocessing extensions."""

from collections.abc import Callable
from typing import Any

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.validation import ConfigError

Loader = Callable[..., BaseDataset]
Transform = Callable[..., BaseDataset]
SplitPolicy = Callable[..., dict[str, Any]]

LOADERS: dict[str, Loader] = {}
TRANSFORMS: dict[str, Transform] = {}
SPLIT_POLICIES: dict[str, SplitPolicy] = {}
```

## Registration pseudocode

```python
def register_loader(name: str, fn: Loader, *, replace: bool = False) -> None:
    _register(LOADERS, kind="loader", name=name, fn=fn, replace=replace)


def register_transform(name: str, fn: Transform, *, replace: bool = False) -> None:
    _register(TRANSFORMS, kind="transform", name=name, fn=fn, replace=replace)


def register_split_policy(name: str, fn: SplitPolicy, *, replace: bool = False) -> None:
    _register(SPLIT_POLICIES, kind="split policy", name=name, fn=fn, replace=replace)


def _register(registry: dict[str, Callable[..., object]], *, kind: str, name: str, fn, replace: bool) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"Cannot register {kind} with blank name")
    if not callable(fn):
        raise ConfigError(f"Registered {kind} {name!r} must be callable")
    if name in registry and not replace:
        raise ConfigError(f"{kind.title()} {name!r} is already registered")
    registry[name] = fn
```

## Lookup pseudocode

```python
def get_loader(name: str) -> Loader:
    return _get(LOADERS, kind="loader", name=name)


def get_transform(name: str) -> Transform:
    return _get(TRANSFORMS, kind="transform", name=name)


def get_split_policy(name: str) -> SplitPolicy:
    return _get(SPLIT_POLICIES, kind="split policy", name=name)


def _get(registry, *, kind: str, name: str):
    try:
        return registry[name]
    except KeyError as exc:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ConfigError(f"Unknown {kind} {name!r}. Available {kind}s: {available}") from exc
```

## Built-in registration

Use one explicit bootstrap function and call it automatically during package import so `load_dataset_bundle()` works out of the box with built-in loaders, transforms, and split policies.

```python
_BUILTINS_REGISTERED = False


def register_builtins(*, replace: bool = False) -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED and not replace:
        return

    from preprocessing_core.loaders.npz import load_npz
    from preprocessing_core.loaders.npy import load_npy
    from preprocessing_core.loaders.csv import load_csv
    from preprocessing_core.splits.chronological import chronological_split
    from preprocessing_core.splits.ranges import index_ranges_split
    from preprocessing_core.splits.seasonal import seasonal_block_split
    from preprocessing_core.transforms.basic import fill_nan

    register_loader("npz", load_npz, replace=replace)
    register_loader("npy", load_npy, replace=replace)
    register_loader("csv", load_csv, replace=replace)
    register_split_policy("chronological", chronological_split, replace=replace)
    register_split_policy("index_ranges", index_ranges_split, replace=replace)
    register_split_policy("seasonal_block", seasonal_block_split, replace=replace)
    register_transform("fill_nan", fill_nan, replace=replace)

    _BUILTINS_REGISTERED = True
```

Package import should call `register_builtins()` once, while keeping optional-dependency imports guarded so the base package import does not fail just because an optional loader dependency is absent. Private repositories then add their own behavior explicitly with `register_loader(...)`, `register_transform(...)`, and `register_split_policy(...)` in ordinary Python setup code.

## Human readability

- No plugin manager.
- No entry-point discovery in v1.
- No dynamic import strings from YAML.
- Keep error messages listing available names.

## Error handling

Use `ConfigError` for registration and lookup failures because users fix these through config or setup code.

## Logging

No logging required. Registration should be quiet.

## Tests

- register and lookup each kind
- duplicate fails unless `replace=True`
- blank name fails
- non-callable fails
- unknown name lists available names
- optional dependency loaders do not break base import
