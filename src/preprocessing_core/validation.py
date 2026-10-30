"""Validation helpers and framework exception types."""

from __future__ import annotations

from typing import Any

import numpy as np

VALID_SPLITS = ("train", "val", "test")


class ConfigError(ValueError):
    """Configuration is invalid and should be fixed by editing config/setup."""


class PipelineError(RuntimeError):
    """Pipeline execution failed after configuration was accepted."""


def validate_dataset(dataset: Any) -> None:
    """Validate the core dataset shape and alignment contract."""
    data = dataset.data
    if not isinstance(data, np.ndarray):
        raise PipelineError("dataset.data must be a NumPy array")
    if data.ndim != 3:
        raise PipelineError(f"dataset.data must have shape (T, C, F); got {data.shape}")

    n_rows, n_channels, n_features = data.shape
    if n_rows == 0:
        raise PipelineError("dataset.data has zero time rows")

    if dataset.timestamps is not None:
        _require_shape(dataset.timestamps, (n_rows,), "timestamps")
    if dataset.labels is not None:
        _require_shape(dataset.labels, (n_rows,), "labels")
    if dataset.channel_labels is not None:
        _require_shape(dataset.channel_labels, (n_rows, n_channels), "channel_labels")
    if dataset.channel_names is not None and len(dataset.channel_names) != n_channels:
        raise PipelineError(
            f"channel_names length {len(dataset.channel_names)} does not match C={n_channels}"
        )
    if dataset.feature_names is not None and len(dataset.feature_names) != n_features:
        raise PipelineError(
            f"feature_names length {len(dataset.feature_names)} does not match F={n_features}"
        )

    aligned_time = dataset.metadata.get("aligned_time", {})
    if aligned_time is None:
        return
    if not isinstance(aligned_time, dict):
        raise PipelineError("metadata['aligned_time'] must be a mapping")
    for key, value in aligned_time.items():
        if not hasattr(value, "shape"):
            raise PipelineError(f"metadata['aligned_time'][{key!r}] must be array-like")
        if value.shape[0] != n_rows:
            raise PipelineError(
                f"metadata['aligned_time'][{key!r}] first dimension {value.shape[0]} "
                f"does not match T={n_rows}"
            )


def validate_apply_and_fit(
    *,
    stage: str,
    name: str,
    apply: tuple[str, ...],
    fit: str | None,
) -> None:
    """Validate transform split scheduling fields."""
    if not apply:
        raise ConfigError(f"{stage} transform {name!r} apply must not be empty")

    unknown = set(apply) - set(VALID_SPLITS)
    if unknown:
        raise ConfigError(f"{stage} transform {name!r} has invalid apply splits: {sorted(unknown)}")
    if len(set(apply)) != len(apply):
        raise ConfigError(f"{stage} transform {name!r} has duplicate apply splits")
    if fit is not None:
        if fit != "train":
            raise ConfigError(f"{stage} transform {name!r} fit must be 'train', got {fit!r}")
        if "train" not in apply:
            raise ConfigError(
                f"{stage} transform {name!r} has fit: train but does not apply to train"
            )


def validate_split_indices(
    indices: dict[str, np.ndarray],
    *,
    n_rows: int,
    chronological_expected: bool = False,
) -> None:
    """Validate split-policy output before materializing datasets."""
    if set(indices) != set(VALID_SPLITS):
        raise ConfigError(f"Split policy must return exactly {VALID_SPLITS}; got {sorted(indices)}")

    seen: set[int] = set()
    for split_name in VALID_SPLITS:
        values = np.asarray(indices[split_name])
        if values.ndim != 1:
            raise ConfigError(f"Split {split_name!r} indices must be 1D")
        if not np.issubdtype(values.dtype, np.integer):
            raise ConfigError(f"Split {split_name!r} indices must be integers")
        if values.size == 0:
            raise ConfigError(f"Split {split_name!r} is empty")
        if np.any(values < 0) or np.any(values >= n_rows):
            raise ConfigError(f"Split {split_name!r} contains indices outside [0, {n_rows})")
        if len(np.unique(values)) != values.size:
            raise ConfigError(f"Split {split_name!r} contains duplicate indices")
        if chronological_expected and np.any(np.diff(values) < 0):
            raise ConfigError(f"Split {split_name!r} indices must be ordered")

        overlap = seen.intersection(map(int, values))
        if overlap:
            raise ConfigError(f"Split {split_name!r} overlaps with an earlier split")
        seen.update(map(int, values))


def validate_state_is_serializable(value: object, *, context: str) -> None:
    """Raise if fitted state cannot be represented as plain serializable data."""
    try:
        _validate_serializable_value(value)
    except TypeError as exc:
        raise PipelineError(f"{context} contains non-serializable fitted state") from exc


def _require_shape(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    array = np.asarray(value)
    if array.shape != shape:
        raise PipelineError(f"{name} must have shape {shape}; got {array.shape}")


def _validate_serializable_value(value: object) -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, np.generic):
        return
    if isinstance(value, np.ndarray):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_serializable_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("state dict keys must be strings")
            _validate_serializable_value(item)
        return
    raise TypeError(f"unsupported state value: {type(value).__name__}")
