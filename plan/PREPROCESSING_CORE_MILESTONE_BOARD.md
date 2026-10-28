# Preprocessing Core + ANLDQ Migration Milestone Board

This board tracks two separate streams:

1. **Standalone core implementation**: build `preprocessing_core` as a public, generic package.
2. **ANLDQ adoption**: migrate the existing `CrossExperimentalAIDQM/src/anldq` stack onto the new contract safely.

The standalone core can begin first. Existing `anldq` adoption is intentionally a later migration track because current CLI, infer, baselines, analysis, and deep-learning paths still depend on legacy dataset fields and shape behavior.

## Board columns

Use these columns in an issue board:

1. Backlog
2. Ready
3. In Progress
4. Blocked
5. Review
6. Done

## Suggested labels

- `core`
- `migration`
- `docs`
- `tests`
- `adoption-risk`
- `dataset-contract`
- `shape-semantics`
- `cli`
- `infer`
- `baselines`
- `analysis`
- `deep-learning`
- `plugin`
- `blocker`

---

# Phase 0: Project framing

Goal: start from one clear source of truth and avoid mixing core implementation with adoption work.

## M0.1: Declare source of truth

**Labels:** `core`, `docs`, `blocker`

### Task

Declare that:

- `MINIMAL_PREPROCESSING_CORE_PLAN.md` is the normative design.
- `migration_guide/*` is implementation guidance.
- `migration_guide/00_plan_edit_suggestions.md` is historical/contextual unless explicitly promoted.

### Acceptance criteria

- A short note exists in the plan or `migration_guide/README.md`.
- Future implementers know which document wins if guidance disagrees.
- No implementation starts before this is resolved.

## M0.2: Create separate adoption backlog

**Labels:** `migration`, `adoption-risk`, `blocker`

### Task

Create explicit follow-on issues for existing `anldq` adoption.

### Acceptance criteria

Issues exist for:

- CLI adoption
- inference adoption
- baseline adoption
- analysis adoption
- deep-learning/input-shape adoption
- real plugin/custom loader proof

These issues are clearly outside standalone core v1 completion.

---

# Phase 1: Core package foundation

Goal: create the package skeleton and public contracts.

## M1.1: Package skeleton and public API

**Labels:** `core`

### Files

- `src/preprocessing_core/__init__.py`
- `src/preprocessing_core/api.py`

### Tasks

Expose:

- `load_dataset_bundle`
- `load_config`
- `build_bundle`
- `register_loader`
- `register_transform`
- `register_split_policy`
- `BaseDataset`
- `DatasetBundle`

### Depends on

- M0.1

### Acceptance criteria

- Package imports cleanly.
- Public API matches the plan.
- Importing `preprocessing_core` does not require optional heavy dependencies such as `h5py`.
- Public functions/classes have concise docstrings.

## M1.2: Dataset contract

**Labels:** `core`, `dataset-contract`

### Files

- `src/preprocessing_core/datasets.py`

### Tasks

Implement:

- `BaseDataset`
- `DatasetBundle`
- `BaseDataset.validate()`
- `BaseDataset.copy()`
- `BaseDataset.with_split()`
- derived properties: `shape`, `n_channels`, `n_features`, `__len__`

### Depends on

- M1.1

### Acceptance criteria

- `BaseDataset` is a plain dataclass.
- It does not load, split, transform, scale, or evaluate.
- `copy()` prevents accidental array/metadata aliasing.
- `with_split()` only tags split metadata.
- No imports from `anldq`.

## M1.3: Registry system

**Labels:** `core`

### Files

- `src/preprocessing_core/registry.py`

### Tasks

Implement registries for:

- loaders
- transforms
- split policies

### Depends on

- M1.1

### Acceptance criteria

- Register/get works for all registry types.
- Duplicate names fail by default.
- `replace=True` works.
- Blank names fail.
- Non-callables fail.
- Unknown-name errors list available names.
- Tests can isolate or restore registry state.

---

# Phase 2: Config and validation

Goal: parse YAML once, apply defaults once, and reject invalid configs early.

## M2.1: Config dataclasses and parser

**Labels:** `core`, `tests`

### Files

- `src/preprocessing_core/config.py`

### Tasks

Implement:

- `PipelineConfig`
- `LoaderConfig`
- `TransformConfig`
- `SplitConfig`
- `load_config(path)`
- `parse_config_dict(raw)`
- strict top-level key validation
- transform field validation
- framework defaults

### Depends on

- M1.1

### Acceptance criteria

- Valid minimal config parses.
- Unknown top-level keys fail.
- Unknown transform fields fail.
- Missing `loader` fails.
- Missing `split` fails.
- Unsupported version fails.
- `pre_split` rejects `apply`.
- `pre_split` rejects `fit`.
- post-split `fit` must be `train`.
- post-split `fit: train` must be included in `apply`.

## M2.2: Validation helpers and errors

**Labels:** `core`, `tests`

### Files

- `src/preprocessing_core/validation.py`

### Tasks

Implement:

- `ConfigError`
- `PipelineError`
- `validate_dataset`
- `validate_apply_and_fit`
- `validate_split_indices`
- `validate_state_is_serializable`

### Depends on

- M1.2
- M2.1

### Acceptance criteria

- Invalid dataset shapes fail clearly.
- Misaligned timestamps, labels, channel labels, channel names, and feature names fail.
- Invalid split indices fail.
- Overlapping splits fail.
- Empty splits fail.
- Non-serializable fitted state fails.
- Error messages name the bad field/config/split where practical.

---

# Phase 3: Selection and alignment

Goal: centralize row/channel slicing so transforms cannot silently desynchronize metadata.

## M3.1: Time and channel selection helpers

**Labels:** `core`, `dataset-contract`, `tests`

### Files

- `src/preprocessing_core/selection.py`

### Tasks

Implement:

- `select_time_rows(dataset, keep_or_indices)`
- `select_channels(dataset, keep_or_indices)`
- optional `select_features(dataset, keep_or_indices)` if feature slicing becomes repeated

### Depends on

- M1.2
- M2.2

### Acceptance criteria

`select_time_rows` slices:

- `data`
- `timestamps`
- `labels`
- `channel_labels`
- every array under `metadata["aligned_time"]`

`select_channels` slices:

- `data`
- `channel_names`
- channel axis of `channel_labels`

General criteria:

- Output validates.
- Source dataset is not accidentally mutated unless explicitly documented.
- Tests prove aligned metadata stays aligned.

---

# Phase 4: Built-in loaders

Goal: support small generic public formats without proprietary dependencies.

## M4.1: Loader package registration

**Labels:** `core`

### Files

- `src/preprocessing_core/loaders/__init__.py`

### Depends on

- M1.3

### Acceptance criteria

- Built-in loader registration works.
- `import preprocessing_core` does not require `h5py`.
- Optional dependencies are imported lazily.

## M4.2: NPY loader

**Labels:** `core`, `tests`

### Files

- `src/preprocessing_core/loaders/npy.py`

### Acceptance criteria

- Loads 1D, 2D, and 3D arrays.
- Normalizes to `(T, C, F)`.
- Optional timestamps/labels/channel labels work.
- Output validates.

## M4.3: NPZ loader

**Labels:** `core`, `tests`

### Files

- `src/preprocessing_core/loaders/npz.py`

### Acceptance criteria

- Loads configured `data_key`.
- Optional `timestamps_key`, `labels_key`, and `channel_labels_key` work.
- Shape normalization works.
- Missing required data key fails clearly.

## M4.4: CSV loader

**Labels:** `core`, `tests`

### Files

- `src/preprocessing_core/loaders/csv.py`

### Acceptance criteria

- Value columns load to `(T, C, 1)`.
- Optional timestamp column works.
- Optional label column works.
- Channel names come from value columns.
- Bad columns fail clearly.

## M4.5: HDF5 loader

**Labels:** `core`, `tests`

### Files

- `src/preprocessing_core/loaders/hdf5.py`

### Acceptance criteria

- `h5py` is imported lazily.
- Data key is configurable.
- Optional timestamp/label/channel-label keys work.
- No complex group hierarchy support in v1.
- Missing optional `h5py` gives a helpful error only when HDF5 loader is used.

---

# Phase 5: Built-in transforms

Goal: implement small generic transforms with strict alignment behavior.

## M5.1: Transform package registration

**Labels:** `core`

### Files

- `src/preprocessing_core/transforms/__init__.py`

### Acceptance criteria

- Built-in transform names are registered.
- Duplicate registration rules are respected.
- Import behavior is predictable.

## M5.2: Basic transforms

**Labels:** `core`, `tests`

### Files

- `src/preprocessing_core/transforms/basic.py`

### Transforms

- `fill_nan`
- `interpolate_nan_per_channel`
- `keep_channels`
- `drop_channels`
- `keep_features`
- `drop_features`
- `validate_monotonic_timestamps`

### Acceptance criteria

- All transforms accept `dataset`, `state`, `fit`, and params.
- All return `BaseDataset`.
- Channel transforms use `select_channels`.
- Feature transforms update `feature_names`.
- Timestamp validation fails clearly.
- Transforms do not touch orchestrator-owned fields.

## M5.3: Filtering transforms

**Labels:** `core`, `tests`

### Files

- `src/preprocessing_core/transforms/filtering.py`

### Acceptance criteria

- Row filtering uses `select_time_rows`.
- Aligned fields stay aligned.
- Invalid quantiles fail.
- Removing all rows fails.
- Transform state remains serializable.

## M5.4: Scaling transforms

**Labels:** `core`, `tests`

### Files

- `src/preprocessing_core/transforms/scaling.py`

### Transforms

- `standard_scale`
- `robust_scale`

### Acceptance criteria

- `fit=True` writes serializable state.
- `fit=False` reuses existing state.
- Missing state fails clearly.
- No sklearn estimator objects are stored.
- `per_channel_feature` mode works.

---

# Phase 6: Split policies

Goal: create deterministic split policies that return indices only.

## M6.1: Split package registration

**Labels:** `core`

### Files

- `src/preprocessing_core/splits/__init__.py`

### Acceptance criteria

- Built-in split policies are registered.
- Unknown split policy errors are helpful.

## M6.2: Chronological split

**Labels:** `core`, `tests`

### Files

- `src/preprocessing_core/splits/chronological.py`

### Acceptance criteria

- Ratios produce contiguous train/val/test.
- `gap=0` keeps adjacent boundaries.
- `gap>0` drops expected boundary rows.
- Invalid ratios fail.
- Too-large gap fails because a split becomes empty.
- Indices are ordered.

## M6.3: Explicit index ranges split

**Labels:** `core`, `tests`

### Files

- `src/preprocessing_core/splits/ranges.py`

### Acceptance criteria

- Half-open ranges work.
- Coverage gaps are allowed.
- Overlaps fail.
- Out-of-range indices fail.
- Empty split fails.
- Gap trimming works if included in v1.

## M6.4: Seasonal block split

**Labels:** `core`, `tests`

### Files

- `src/preprocessing_core/splits/seasonal.py`

### Acceptance criteria

- Requires timestamps.
- Converts timestamps to calendar year/month.
- Assigns contiguous blocks by year/season strata.
- Uses `np.random.default_rng(seed)`.
- Does not use Python `hash()` for reproducibility.
- Gap trimming works.
- Missing/bad timestamps fail clearly.

---

# Phase 7: Pipeline orchestration

Goal: implement the readable one-load end-to-end pipeline.

## M7.1: Pipeline scheduling

**Labels:** `core`, `tests`

### Files

- `src/preprocessing_core/pipeline.py`

### Tasks

Implement:

- `ScheduledStep`
- `schedule_transform_steps`
- callable lookup
- `apply` / `fit` scheduling
- `state_key` generation

### Depends on

- M1.3
- M2.1
- M2.2

### Acceptance criteria

- Pre-split rejects `apply` and `fit`.
- Post-split defaults `apply` to all splits.
- `fit: train` is validated.
- `state_key` defaults predictably.
- Duplicate transform instances get distinct default state keys.

## M7.2: One-load build flow

**Labels:** `core`, `tests`

### Files

- `src/preprocessing_core/pipeline.py`

### Pipeline

```text
load once
-> pre_split transforms once
-> split indices
-> materialize train/val/test
-> post_split transforms
-> DatasetBundle
```

### Depends on

- M3.1
- at least one loader
- at least one transform
- at least one split policy
- M7.1

### Acceptance criteria

- Loader is called exactly once.
- Pre-split transforms run exactly once.
- Split datasets are independent objects.
- Post-split transforms run in train → val → test order.
- `fit: train` state is available to val/test.
- Empty split after transform fails.
- Transform errors include stage/name/split context.
- Misbehaving transforms cannot mutate `split`, `data_tag`, or `preprocess_steps`.

## M7.3: Bundle metadata and fitted state

**Labels:** `core`, `tests`

### Files

- `src/preprocessing_core/pipeline.py`
- helper module only if needed

### Acceptance criteria

Bundle metadata includes:

- config version
- config path
- loader name
- split policy
- split params
- pre/post step names
- final shapes
- final lengths
- dropped-row metadata where available
- fitted state

## M7.4: Public API wrappers

**Labels:** `core`

### Files

- `src/preprocessing_core/api.py`

### Tasks

Implement:

- `build_bundle(config)`
- `load_dataset_bundle(path)`

### Acceptance criteria

- Wrappers are thin.
- No duplicate orchestration logic exists outside `PreprocessingPipeline`.

---

# Phase 8: Core tests, docs, and examples

Goal: make the standalone package independently usable and reviewable.

## M8.1: Test suite

**Labels:** `core`, `tests`

### Files

- `tests/preprocessing_core/conftest.py`
- `tests/preprocessing_core/test_datasets.py`
- `tests/preprocessing_core/test_config.py`
- `tests/preprocessing_core/test_registration.py`
- `tests/preprocessing_core/test_selection.py`
- `tests/preprocessing_core/test_loaders.py`
- `tests/preprocessing_core/test_transforms.py`
- `tests/preprocessing_core/test_splits.py`
- `tests/preprocessing_core/test_pipeline.py`

### Acceptance criteria

Tests prove the known bug classes are fixed:

- repeated loads
- scaler-state leakage
- silent empty splits
- metadata misalignment

## M8.2: Example configs

**Labels:** `core`, `docs`

### Files

- `configs/example_npz.yaml`
- `configs/example_csv.yaml`

### Acceptance criteria

- No absolute paths.
- No proprietary dataset names.
- No private transforms.
- Examples are understandable without reading source.

## M8.3: User docs

**Labels:** `core`, `docs`

### Files

- `docs/preprocessing_core/configuration.md`
- `docs/preprocessing_core/custom_extensions.md`

### Acceptance criteria

Docs explain:

- one-load pipeline
- `(T, C, F)` convention
- YAML schema
- pre-split vs post-split
- `apply` and `fit`
- fitted state
- split gaps
- metadata conventions
- custom registration
- why YAML avoids dotted imports
- plugin author checklist

## M8.4: Core v1 completion review

**Labels:** `core`, `tests`, `docs`

### Acceptance criteria

- Standalone package is implemented.
- Tests pass.
- Docs exist.
- Examples exist.
- No imports from `anldq`.
- No hidden adoption work is counted as complete.

---

# Phase 9: Adoption planning and adapters

Goal: prepare migration without destabilizing current workflows.

## M9.1: Legacy dataset contract audit

**Labels:** `migration`, `dataset-contract`, `adoption-risk`

### Task

Audit current `anldq` dependencies on:

- `raw_data`
- `data`
- `headers`
- `meta`
- `channel_groups`
- `timestamps`
- `labels`
- `fitted_scaler`
- `fitted_baseline_state`
- `split_group_lengths`
- `file_lengths`
- `data_original`
- `n_channels`
- `n_features`

### Acceptance criteria

Each dependency has:

- file path
- current behavior
- replacement strategy
- owner: core, adapter, or downstream code

## M9.2: Compatibility adapter design

**Labels:** `migration`, `dataset-contract`

### Task

Choose one adoption strategy:

1. Adapter class exposing legacy fields from new datasets.
2. Conversion function producing current-style dataset objects.
3. Direct migration of call sites to new fields.

### Depends on

- M9.1

### Acceptance criteria

- One approach is chosen.
- Tradeoffs are documented.
- Adapter lives under `anldq`, not `preprocessing_core`, if it needs legacy behavior.

## M9.3: Real custom loader proof sketch

**Labels:** `plugin`, `migration`, `adoption-risk`

### Task

Sketch one real private/custom loader for GM2, HLT, or SPT.

### Acceptance criteria

The sketch covers:

- source reads
- shape normalization
- timestamps
- labels/channel labels
- channel names
- feature names
- metadata
- quality/channel filtering location
- downstream leftovers

---

# Phase 10: CLI adoption

Goal: optionally let CLI build data through the new core without breaking current workflows.

## M10.1: Decide CLI integration mode

**Labels:** `migration`, `cli`

### Options

1. New CLI flag for preprocessing-core config.
2. New dataset family/resolver path.
3. Internal replacement of current dataset loading.

### Depends on

- M9.2

### Acceptance criteria

- One mode is selected.
- Backward compatibility strategy is explicit.
- Existing CLI configs do not silently change behavior.

## M10.2: CLI bundle-loading path

**Labels:** `migration`, `cli`

### Depends on

- M10.1
- M9.2

### Acceptance criteria

- Existing legacy CLI path still works.
- New core-backed CLI path is opt-in.
- Train/val/test are available to trainer.
- Test inference can receive a test dataset object.

## M10.3: CLI artifact metadata

**Labels:** `migration`, `cli`

### Acceptance criteria

- No direct hard requirement on legacy `headers`.
- New `channel_names` path works.
- Backward compatibility fallback exists if needed.

---

# Phase 11: Infer adoption

Goal: decouple inference from legacy dataset class checks and metadata assumptions.

## M11.1: Replace GM2 class-name check

**Labels:** `migration`, `infer`, `shape-semantics`

### Task

Replace `ds.__class__.__name__ == "GM2Dataset"` with an explicit metadata/config flag.

### Acceptance criteria

- No inference behavior depends on dataset class name.
- GM2-equivalent behavior remains possible.
- Default non-GM2 behavior is unchanged.

## M11.2: Normalize infer metadata access

**Labels:** `migration`, `infer`, `dataset-contract`

### Task

Replace direct assumptions about:

- `meta`
- `headers`
- `channel_groups`

with a compatibility helper or adapter.

### Acceptance criteria

- New datasets can provide `metadata`, `channel_names`, and optional `metadata["channel_groups"]`.
- Legacy datasets still work.
- Infer output names and score aggregation remain stable.

## M11.3: Score aggregation ownership decision

**Labels:** `migration`, `infer`, `analysis`

### Task

Decide where score aggregation belongs:

1. downstream in inference/analysis
2. adapter/result layer
3. formal metadata convention

### Acceptance criteria

- One canonical owner is chosen.
- `preprocessing_core.BaseDataset` does not become an evaluation object.
- Infer and analysis agree on score map semantics.

---

# Phase 12: Baseline adoption

Goal: remove accidental dependency on legacy `raw_data` without changing baseline behavior.

## M12.1: Baseline input contract decision

**Labels:** `migration`, `baselines`

### Options

1. Baselines continue accepting arrays.
2. Baselines accept dataset objects.
3. Runner extracts arrays from dataset objects.

### Acceptance criteria

- One contract is selected.
- Multi-feature behavior is documented.
- Existing baseline outputs remain compatible.

## M12.2: Shared baseline shape adapter

**Labels:** `migration`, `baselines`, `shape-semantics`

### Task

Create one helper for baseline input conversion from `(T, C, F)` to expected baseline views.

### Acceptance criteria

- Repeated `_ensure_tc` logic can be reduced.
- Behavior drift is avoided.
- Each baseline’s feature handling is explicit.

## M12.3: Replace `dataset.raw_data` baseline dependency

**Labels:** `migration`, `baselines`

### Acceptance criteria

- Baseline recommendation works without mandatory `raw_data`.
- Baseline benchmark path works with new datasets or adapter.
- Legacy datasets still work.

---

# Phase 13: Analysis and benchmark adoption

Goal: unify score/label normalization and make preprocessing boundaries explicit.

## M13.1: Choose canonical score-collapse path

**Labels:** `migration`, `analysis`

### Task

Resolve overlap between direct `normalize_scores_to_1d()` behavior and YAML-driven `aggregate_scores()` behavior.

### Acceptance criteria

- One canonical path is documented.
- Benchmark executors and model eval agree.
- Required `time` score behavior is explicit.

## M13.2: Synthetic injection boundary

**Labels:** `migration`, `analysis`

### Task

Define whether anomaly injection happens before preprocessing, after preprocessing, on raw arrays, or on `BaseDataset` copies.

### Acceptance criteria

- Benchmark mutation behavior is explicit.
- Labels remain aligned.
- Injection does not silently bypass preprocessing assumptions.

## M13.3: Analysis metadata adapter

**Labels:** `migration`, `analysis`, `dataset-contract`

### Acceptance criteria

New metadata locations work for:

- timestamps
- channel names
- channel groups
- file/group IDs
- split grouping metadata

---

# Phase 14: Deep-learning and trainer adoption

Goal: make model input shape policy explicit and avoid checkpoint surprises.

## M14.1: Shape policy document

**Labels:** `migration`, `deep-learning`, `shape-semantics`

### Task

Document how `(T, C, F)` becomes model input.

Must cover:

- default feature reduction
- `subsensor_mode`
- flattening order
- channel-major assumptions
- feature-axis preservation
- output reshaping
- forecasting target behavior

### Acceptance criteria

- `AutoModelSelector`, trainer prep, and infer prep can point to one policy.
- GM2-like feature-axis preservation is dataset-agnostic.
- Multi-feature behavior is not implicit.

## M14.2: Trainer/infer data-prep adapter

**Labels:** `migration`, `deep-learning`, `infer`

### Task

Adapt or wrap trainer/infer data prep to consume the new dataset contract or compatibility adapter.

### Acceptance criteria

- Existing models still receive expected shapes.
- New core-backed datasets can train/infer.
- `subsensor_mode` source of truth is clear.

## M14.3: Checkpoint compatibility plan

**Labels:** `migration`, `deep-learning`

### Task

Define what happens when preprocessing changes effective input dims.

### Acceptance criteria

- Old checkpoint behavior is documented.
- New checkpoints record enough preprocessing/shape metadata.
- Incompatible checkpoint loads fail clearly.

---

# Phase 15: Pilot migration

Goal: prove the path with one real dataset family before broad migration.

## M15.1: Select pilot dataset family

**Labels:** `migration`, `plugin`

### Recommended candidates

- HLT for broad current-use coverage
- GM2 for multi-feature/feature-axis behavior
- SPT for stable-channel/custom-loader behavior

### Acceptance criteria

- One pilot is chosen.
- Success criteria are written.
- Non-goals are written.

## M15.2: Implement pilot custom loader/plugin path

**Labels:** `migration`, `plugin`

### Acceptance criteria

- Pilot data can be loaded through `preprocessing_core`.
- Output bundle validates.
- Required metadata is present.
- Proprietary/domain-specific logic does not live in public core.

## M15.3: Pilot train/infer path

**Labels:** `migration`, `cli`, `infer`, `deep-learning`

### Acceptance criteria

- Pilot can train.
- Pilot can run inference.
- Output artifacts are comparable to legacy behavior.
- Behavior differences are documented.

## M15.4: Pilot analysis/baseline path

**Labels:** `migration`, `analysis`, `baselines`

### Acceptance criteria

- Pilot can run baseline or analysis workflow if applicable.
- Score aggregation follows the chosen canonical path.
- Labels/timestamps/channel names remain aligned.

---

# Phase 16: Broader migration and deprecation

Goal: move from pilot to broader adoption safely.

## M16.1: User migration guide

**Labels:** `migration`, `docs`

### Task

Write user-facing migration docs covering:

- legacy dataset config
- new preprocessing config
- field mapping
- behavior changes
- unsupported legacy paths

### Acceptance criteria

- Users can migrate one config with docs alone.
- Known differences are explicit.
- Deprecated fields are listed.

## M16.2: Deprecation warnings

**Labels:** `migration`

### Acceptance criteria

- Warnings are actionable.
- Warnings are not noisy in normal current workflows.
- Deprecation schedule is clear.

## M16.3: Remove or freeze old paths

**Labels:** `migration`

### Acceptance criteria

- Only after real adoption proves stable.
- Legacy paths are removed, frozen, or clearly marked.
- No silent behavior changes.

---

# Recommended execution order

## Start immediately

1. M0.1: Source of truth
2. M0.2: Adoption backlog
3. M1.1: Package skeleton
4. M1.2: Dataset contract
5. M1.3: Registry

## Then parallelize

### Stream A: Core framework

- M2: Config/validation
- M3: Selection
- M7: Pipeline

### Stream B: Built-ins

- M4: Loaders
- M5: Transforms
- M6: Splits

### Stream C: Quality

- M8: Tests/docs/examples

## Start migration planning while core is being built

- M9.1: Legacy contract audit
- M9.2: Adapter design
- M9.3: Real custom loader proof sketch
- M14.1: Shape policy document

## Do not start broad integration until

- Core interfaces are stable.
- Adapter design is complete.
- Shape policy is documented.

---

# Definitions of done

## Core v1 done

Core v1 is done when:

- `preprocessing_core` exists.
- It has no imports from `anldq`.
- Public API works.
- Loaders/transforms/splits are registered.
- Pipeline loads once.
- Pre-split transforms run once.
- Post-split `fit/apply` works.
- Split validation rejects overlap and empty splits.
- Aligned metadata stays aligned.
- Fitted state is serializable and present in bundle metadata.
- Tests cover the known bug classes.
- Example configs and docs exist.

## Migration planning done

Migration planning is done when:

- legacy field dependencies are audited.
- adapter strategy is chosen.
- shape policy is documented.
- score aggregation ownership is decided.
- at least one real custom loader is sketched.

## Pilot migration done

Pilot migration is done when:

- one real dataset family loads through the new core.
- train/infer works or has a documented blocker.
- analysis/baseline path works or has a documented non-goal.
- behavior differences from legacy are known.

---

# Efficiency rule

Do not migrate CLI/infer/baselines/analysis while building the core.

The efficient path is:

1. Build the standalone core cleanly.
2. In parallel, audit and design adapters.
3. Prove one real custom loader.
4. Pilot one dataset family.
5. Broaden adoption only after the pilot is understood.
