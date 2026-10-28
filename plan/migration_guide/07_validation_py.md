# Migration Map: `src/preprocessing_core/validation.py`

## Purpose

Central validation helpers and small public exception types.

This file lets the mini-framework enforce rules consistently without bloating `PreprocessingPipeline`, transforms, or loaders.

## Scope

Owns:

- `ConfigError`
- `PipelineError`
- dataset shape/alignment validation
- split index validation
- `apply` / `fit` validation
- serializable state validation
- ratio validation helpers

Does not own:

- YAML parsing
- pipeline sequencing
- transform-specific math
- split-policy index creation

## Exceptions

```python
"""Validation helpers and framework exception types."""

class ConfigError(ValueError):
    """Configuration is invalid and should be fixed by editing config/setup."""


class PipelineError(RuntimeError):
    """Pipeline execution failed after configuration was accepted."""
```

Do not add a deep hierarchy in v1.

## Dataset validation pseudocode

```python
def validate_dataset(dataset) -> None:
    data = dataset.data
    if not isinstance(data, np.ndarray):
        raise PipelineError("dataset.data must be a NumPy array")
    if data.ndim != 3:
        raise PipelineError(f"dataset.data must have shape (T, C, F); got {data.shape}")
    t, c, f = data.shape
    if t == 0:
        raise PipelineError("dataset.data has zero time rows")

    if dataset.timestamps is not None:
        _require_shape(dataset.timestamps, (t,), "timestamps")
    if dataset.labels is not None:
        _require_shape(dataset.labels, (t,), "labels")
    if dataset.channel_labels is not None:
        _require_shape(dataset.channel_labels, (t, c), "channel_labels")
    if dataset.channel_names is not None and len(dataset.channel_names) != c:
        raise PipelineError(f"channel_names length {len(dataset.channel_names)} does not match C={c}")
    if dataset.feature_names is not None and len(dataset.feature_names) != f:
        raise PipelineError(f"feature_names length {len(dataset.feature_names)} does not match F={f}")

    aligned_time = dataset.metadata.get("aligned_time", {})
    if aligned_time is None:
        return
    if not isinstance(aligned_time, dict):
        raise PipelineError("metadata['aligned_time'] must be a mapping")
    for key, value in aligned_time.items():
        if not hasattr(value, "shape"):
            raise PipelineError(f"metadata['aligned_time'][{key!r}] must be array-like")
        if value.shape[0] != t:
            raise PipelineError(
                f"metadata['aligned_time'][{key!r}] first dimension {value.shape[0]} does not match T={t}"
            )
```

## Split index validation pseudocode

```python
VALID_SPLITS = ("train", "val", "test")


def validate_split_indices(indices: dict[str, np.ndarray], *, n_rows: int, chronological_expected: bool = False) -> None:
    if set(indices) != set(VALID_SPLITS):
        raise ConfigError(f"Split policy must return exactly {VALID_SPLITS}; got {sorted(indices)}")

    seen: set[int] = set()
    for split_name in VALID_SPLITS:
        arr = np.asarray(indices[split_name])
        if arr.ndim != 1:
            raise ConfigError(f"Split {split_name!r} indices must be 1D")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ConfigError(f"Split {split_name!r} indices must be integers")
        if arr.size == 0:
            raise ConfigError(f"Split {split_name!r} is empty")
        if np.any(arr < 0) or np.any(arr >= n_rows):
            raise ConfigError(f"Split {split_name!r} contains indices outside [0, {n_rows})")
        if len(np.unique(arr)) != arr.size:
            raise ConfigError(f"Split {split_name!r} contains duplicate indices")
        if chronological_expected and np.any(np.diff(arr) < 0):
            raise ConfigError(f"Split {split_name!r} indices must be ordered")

        overlap = seen.intersection(map(int, arr))
        if overlap:
            raise ConfigError(f"Split {split_name!r} overlaps with an earlier split")
        seen.update(map(int, arr))
```

## `apply` / `fit` validation

```python
def validate_apply_and_fit(*, stage: str, name: str, apply: tuple[str, ...], fit: str | None) -> None:
    valid = {"train", "val", "test"}
    if not apply:
        raise ConfigError(f"{stage} transform {name!r} apply must not be empty")
    unknown = set(apply) - valid
    if unknown:
        raise ConfigError(f"{stage} transform {name!r} has invalid apply splits: {sorted(unknown)}")
    if len(set(apply)) != len(apply):
        raise ConfigError(f"{stage} transform {name!r} has duplicate apply splits")
    if fit is not None:
        if fit != "train":
            raise ConfigError(f"{stage} transform {name!r} fit must be 'train', got {fit!r}")
        if "train" not in apply:
            raise ConfigError(f"{stage} transform {name!r} has fit: train but does not apply to train")
```

## Serializable state validation

```python
def validate_state_is_serializable(value: object, *, context: str) -> None:
    try:
        _to_jsonable_or_np(value)
    except TypeError as exc:
        raise PipelineError(f"{context} contains non-serializable fitted state") from exc
```

Allowed:

- `None`
- `bool`, `int`, `float`, `str`
- lists/tuples/dicts of allowed values
- NumPy arrays
- NumPy scalar values

Rejected:

- sklearn estimator objects
- open files
- callables
- arbitrary class instances

## Human readability

- Keep validators focused and named by what they validate.
- Error messages should include the field/split/transform name.
- Do not hide validation inside large pipeline methods.

## Logging

No logging here. Validators raise.

## Tests

- dataset validation for every aligned field
- split validation rejects missing names, overlap, duplicates, out-of-range, empty
- `apply`/`fit` validation cases
- non-serializable fitted state fails
- NumPy state passes
