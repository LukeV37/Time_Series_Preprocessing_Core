# Migration Map: Tests, Example Configs, and Docs

## Purpose

Tests and docs are part of the framework, not cleanup. They teach future users and agents the intended rules.

## Test layout

Planned location:

```text
tests/preprocessing_core/
  conftest.py
  test_datasets.py
  test_config.py
  test_pipeline.py
  test_transforms.py
  test_splits.py
  test_registration.py
```

## `conftest.py`

### Scope

- tiny synthetic arrays
- temporary `.npz`/`.npy`/CSV files
- call-count loader fixtures
- registry cleanup fixture if global registries are used

### Pseudocode

```python
@pytest.fixture
def tiny_dataset() -> BaseDataset:
    data = np.arange(30, dtype=float).reshape(10, 3, 1)
    return BaseDataset(
        data=data,
        timestamps=np.arange(10),
        labels=np.zeros(10),
        channel_labels=np.zeros((10, 3)),
        channel_names=["a", "b", "c"],
        feature_names=["value"],
        metadata={"aligned_time": {"file_id": np.zeros(10, dtype=int)}},
    )


@pytest.fixture(autouse=True)
def isolated_registries():
    snapshot = copy_registries()
    try:
        yield
    finally:
        restore_registries(snapshot)
```

If the production registry does not expose copy/restore, tests can use `replace=True` carefully or clear internal registries through test-only helpers.

---

## `test_datasets.py`

Must cover:

- valid `(T, C, F)` dataset passes
- `(T,)` raw data is not accepted directly by `BaseDataset`; loaders normalize it first
- timestamp length mismatch fails
- labels length mismatch fails
- channel labels shape mismatch fails
- channel names length mismatch fails
- feature names length mismatch fails
- `metadata['aligned_time']` first-dimension mismatch fails
- `copy()` prevents aliasing
- `with_split()` sets split/data_tag without changing source

---

## `test_config.py`

Must cover:

- valid minimal YAML parses
- defaults are applied
- unknown top-level key fails
- unknown transform key fails
- missing loader fails
- missing split fails
- unsupported version fails
- `pre_split` rejects `apply`
- `pre_split` rejects `fit`
- post-split `apply` must be non-empty subset of `train`, `val`, `test`
- post-split `fit` must be `train`
- `fit: train` must be included in `apply`

Error messages should mention the exact field or transform.

---

## `test_registration.py`

Must cover:

- register/get loader
- register/get transform
- register/get split policy
- duplicate name fails by default
- `replace=True` works for tests
- blank name fails
- non-callable fails
- unknown name error lists available names
- importing `preprocessing_core` does not require optional `h5py`

---

## `test_splits.py`

Must cover `chronological`:

- deterministic contiguous boundaries
- floor train/val and remainder to test
- `gap=0` no drops
- `gap>0` removes expected boundary rows
- invalid ratios fail
- too-large gap fails because a split becomes empty
- no overlap

Must cover `index_ranges`:

- half-open ranges work
- coverage gaps allowed
- overlap fails
- out-of-range fails
- malformed range fails
- gap trimming works
- empty result fails

---

## `test_transforms.py`

Must cover basic transforms:

- `fill_nan` replaces NaNs and preserves metadata
- `interpolate_nan_per_channel` handles simple NaNs
- `keep_channels` by names and indices
- `drop_channels` by names and indices
- `keep_features` / `drop_features`
- `validate_monotonic_timestamps` success/failure

Must cover filtering:

- `filter_quantile_timesteps` slices all time-aligned fields
- all rows removed fails
- invalid quantiles fail

Must cover scaling:

- `standard_scale` with `fit=True` writes mean/std
- val/test with `fit=False` use existing state
- val/test fail if state missing
- `robust_scale` same pattern
- state contains serializable values only

Must cover orchestrator-owned fields:

- any transform that modifies `split`, `data_tag`, or `preprocess_steps` fails when run through pipeline helper

---

## `test_pipeline.py`

This is the critical mini-framework acceptance test.

### Required end-to-end fixture

Create a tiny `.npz` with:

- `data` shape `(T, C, F)` or `(T, C)`
- `timestamps`
- optional `labels`
- channel names if practical

### Required pipeline config

```yaml
version: 1
loader:
  type: counting_npz
  params:
    path: tiny.npz
split:
  policy: chronological
  params:
    ratios: [0.6, 0.2, 0.2]
    gap: 1
pre_split:
  - name: fill_nan
    params:
      value: 0.0
post_split:
  - name: filter_quantile_timesteps
    apply: [train, val]
    params:
      low_quantile: 0.0
      high_quantile: 100.0
      min_valid_fraction: 1.0
  - name: standard_scale
    fit: train
    apply: [train, val, test]
```

### Assertions

- loader called exactly once
- pre-split transform called exactly once
- split datasets have split names set
- data_tag comes from metadata dataset_name when configured
- post-split transform order is train -> val -> test per step
- val/test scaling uses train state
- bundle metadata includes:
  - config version
  - loader name
  - split policy
  - final shapes
  - final lengths
  - pre/post step names
  - fitted state
- errors from a failing transform include stage/name/split context

---

## Example configs

Planned location:

```text
configs/example_npz.yaml
configs/example_csv.yaml
```

Rules:

- no absolute paths
- no proprietary dataset names
- no private transforms
- examples should be runnable after generating tiny local data or using a checked-in tiny fixture
- comments should be concise and explain scheduling rules

---

## Docs

Planned location:

```text
docs/preprocessing_core/configuration.md
docs/preprocessing_core/custom_extensions.md
```

## `configuration.md`

Must explain:

- one-load pipeline diagram
- `(T, C, F)` convention
- YAML schema
- framework defaults
- pre-split vs post-split semantics
- `apply` and `fit`
- fitted state
- split gaps
- metadata/provenance
- error behavior
- logging policy

## `custom_extensions.md`

Must explain:

- loader contract
- transform contract
- split policy contract
- registration examples
- why YAML does not use dotted imports
- how to decompose complex private loaders
- how to avoid alignment bugs
- how to test a plugin

### Plugin author checklist

```markdown
A custom loader should:
- return valid `BaseDataset`
- normalize to `(T, C, F)`
- not split data
- not perform public transform scheduling
- decompose complex source-specific work into helpers

A custom transform should:
- accept/return `BaseDataset`
- use `select_time_rows`/`select_channels`
- store only serializable state
- avoid touching orchestrator-owned fields

A custom split policy should:
- return train/val/test index arrays only
- avoid overlapping splits
- support deterministic behavior
- use `np.random.default_rng(seed)` if random
```

---

## Logging tests

Use `caplog` to verify:

- build start and finish logged at INFO
- loader/split policy logged at INFO
- transform execution logged at DEBUG when enabled
- no `print()` output from library code

Do not assert exact wording too rigidly unless needed. Assert useful fields are present.

---

## Human readability acceptance criteria

Before marking implementation complete, check:

- `PreprocessingPipeline.build()` reads top-to-bottom like the pipeline diagram
- no file contains copied legacy `anldq` logic wholesale
- public modules have short module docstrings
- public functions/classes have concise docstrings
- error messages name the bad field/step/split
- config examples are understandable without reading source code
- each test name describes the behavior it protects
