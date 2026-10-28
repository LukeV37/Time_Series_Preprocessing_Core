# Migration Map: `src/preprocessing_core/transforms/basic.py`

## Purpose

Generic, public transforms that are safe before splitting because they do not fit train-specific state.

## Scope

Owns:

- NaN filling/interpolation
- deterministic channel selection
- deterministic feature selection
- timestamp monotonicity validation

Does not own:

- train-fitted scaling
- row filtering based on split-specific statistics
- domain-specific quality selection

## Transform signature

Every transform must accept:

```python
def transform(dataset: BaseDataset, *, state: dict[str, object], fit: bool, **params) -> BaseDataset:
    ...
```

Even if `state` and `fit` are unused.

## Pseudocode examples

### `fill_nan`

```python
def fill_nan(dataset: BaseDataset, *, state: dict[str, object], fit: bool, value: float = 0.0) -> BaseDataset:
    """Replace NaN values in data with a constant."""
    result = dataset.copy()
    result.data = np.nan_to_num(result.data, nan=value)
    result.validate()
    return result
```

### `validate_monotonic_timestamps`

```python
def validate_monotonic_timestamps(dataset: BaseDataset, *, state: dict[str, object], fit: bool, strict: bool = True) -> BaseDataset:
    """Raise if timestamps are not monotonic."""
    if dataset.timestamps is None:
        return dataset
    diffs = np.diff(dataset.timestamps)
    ok = np.all(diffs > 0) if strict else np.all(diffs >= 0)
    if not ok:
        raise PipelineError("timestamps must be strictly increasing" if strict else "timestamps must be non-decreasing")
    return dataset
```

### `keep_channels`

```python
def keep_channels(dataset: BaseDataset, *, state: dict[str, object], fit: bool, names: list[str] | None = None, indices: list[int] | None = None) -> BaseDataset:
    """Keep selected channels by name or index."""
    if (names is None) == (indices is None):
        raise ConfigError("keep_channels requires exactly one of names or indices")
    if names is not None:
        if dataset.channel_names is None:
            raise PipelineError("keep_channels by name requires channel_names")
        index_by_name = {name: i for i, name in enumerate(dataset.channel_names)}
        missing = [name for name in names if name not in index_by_name]
        if missing:
            raise ConfigError(f"Unknown channel names: {missing}")
        indices = [index_by_name[name] for name in names]
    return select_channels(dataset, indices)
```

### `drop_channels`

```python
def drop_channels(...):
    keep = np.ones(dataset.n_channels, dtype=bool)
    keep[drop_indices] = False
    return select_channels(dataset, keep)
```

### `keep_features` / `drop_features`

Feature selection updates:

- `data[:, :, feature_selector]`
- `feature_names`

No dedicated helper is required in v1 because no other field is feature-aligned.

## Error handling

- Use `ConfigError` for invalid transform parameters.
- Use `PipelineError` for missing dataset fields required by transform behavior.
- Do not silently ignore unknown channel/feature names.

## Logging

No logging in transforms. Pipeline logs transform stage and resulting shape.

## Human readability

- One transform, one behavior.
- Avoid combo transforms.
- Always return a `BaseDataset`.
- Prefer `dataset.copy()` unless transform has a strong reason not to.

## Tests

- each transform preserves dataset validity
- keep/drop channels updates `channel_labels` and `channel_names`
- keep/drop features updates `feature_names`
- monotonic timestamp validation fails clearly
- transforms do not alter orchestrator-owned fields
