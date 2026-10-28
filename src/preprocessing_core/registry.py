"""Explicit callable registries for preprocessing extensions."""

from __future__ import annotations

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


def register_loader(name: str, fn: Loader, *, replace: bool = False) -> None:
    """Register a loader under a short config name."""
    _register(LOADERS, kind="loader", name=name, fn=fn, replace=replace)


def register_transform(name: str, fn: Transform, *, replace: bool = False) -> None:
    """Register a transform under a short config name."""
    _register(TRANSFORMS, kind="transform", name=name, fn=fn, replace=replace)


def register_split_policy(name: str, fn: SplitPolicy, *, replace: bool = False) -> None:
    """Register a split policy under a short config name."""
    _register(SPLIT_POLICIES, kind="split policy", name=name, fn=fn, replace=replace)


def get_loader(name: str) -> Loader:
    """Return a registered loader."""
    return _get(LOADERS, kind="loader", name=name)


def get_transform(name: str) -> Transform:
    """Return a registered transform."""
    return _get(TRANSFORMS, kind="transform", name=name)


def get_split_policy(name: str) -> SplitPolicy:
    """Return a registered split policy."""
    return _get(SPLIT_POLICIES, kind="split policy", name=name)


def _register(
    registry: dict[str, Callable[..., object]],
    *,
    kind: str,
    name: str,
    fn: Callable[..., object],
    replace: bool,
) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"Cannot register {kind} with blank name")
    if not callable(fn):
        raise ConfigError(f"Registered {kind} {name!r} must be callable")
    if name in registry and not replace:
        raise ConfigError(f"{kind.title()} {name!r} is already registered")
    registry[name] = fn


def _get(registry: dict[str, Callable[..., object]], *, kind: str, name: str):
    try:
        return registry[name]
    except KeyError as exc:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ConfigError(f"Unknown {kind} {name!r}. Available {kind}s: {available}") from exc
