# Minimal YAML-Configured Preprocessing Core: Feature Plan

## 1. Purpose and scope

Create a **new, standalone preprocessing package** inside this repository, intended eventually to be extractable into its own public package with zero proprietary dependencies. For now it is implemented as a new subpackage (e.g. `src/preprocessing_core/`) and does not touch or import anything under `src/anldq/`.

The package must:

- load one complete raw dataset through a configured loader;
- use one canonical in-memory data representation: **`(time, channel, feature)`**, abbreviated `(T, C, F)`;
- keep time-aligned timestamps, optional per-timestep labels, and optional per-timestep-per-channel labels aligned whenever rows are filtered or split;
- run YAML-configured transforms before and after the split;
- split the processed full data once according to a YAML-selected split policy;
- support transforms that apply only to selected splits, for example `apply: [train, val]`;
- support transforms that fit state on train and reuse that state for val/test;
- return `DatasetBundle(train, val, test, metadata)` whose `train`/`val`/`test` fields are a new, independent `BaseDataset` class that can be passed directly to the existing `anldq` trainer/infer code with no adapter layer — it carries every field that code's default path hard-requires or reads for behavior (not every field the current `anldq.datasets.base.BaseDataset` happens to have; see the audit in §5);
- let experiments plug in their own loaders, transforms, and split policies without putting proprietary code in the public package;
- persist fitted preprocessing state (e.g. a scaler's mean/std) so it can be reused outside the immediate `build()` call (saved to disk and/or carried in bundle metadata), not just held in a throwaway in-memory dict.

The package must **not** include:

- model training, inference, evaluation, anomaly scoring, plotting, or CLI orchestration;
- proprietary data loaders, proprietary paths, proprietary dataset names, or proprietary dependencies;
- a class hierarchy — the new `BaseDataset` is a standalone, non-subclassed dataclass with a distinct import path (`preprocessing_core.datasets.BaseDataset`) and does not subclass or get subclassed by the existing `anldq.datasets.base.BaseDataset`. It is designed from the start to be the object that `anldq` trainer/infer code consumes (see §5), not an unrelated parallel type;
- repeated loading of the same raw source merely to construct train/val/test splits;
- Parquet loaders or sophisticated split policies in v1 — these are deliberately deferred (see §9, §12) since real production split policies belong in a private plugin repository. v1 ships a small set of generic loaders (including a minimal HDF5 loader, §6.1) and split policies sufficient to validate the architecture end to end and to support real-format inputs out of the box.

This plan is a new minimal core, developed incrementally inside this repo, not a direct copy of the current `anldq.datasets` implementation.

### 1.1 Explicit adoption boundary

This plan's implementation target is **the standalone preprocessing core itself**: config parsing, registration, loading, pre-split transforms, split policies, post-split transforms, `BaseDataset`, and `DatasetBundle`.

It does **not** claim that the rest of the current `CrossExperimentalAIDQM/src/anldq/` stack can consume the new objects unchanged on day one. Current shipped code outside preprocessing still depends on several legacy dataset contracts that this plan intentionally does not preserve verbatim, including:

- tuple-returning dataset-loading entry points rather than a `DatasetBundle` object;
- legacy dataset attribute names such as `meta` and `headers` rather than `metadata` and `channel_names`;
- direct reads of `channel_groups` as a top-level dataset attribute rather than from `metadata`;
- a GM2-specific class-name check in inference code rather than an explicit metadata flag;
- downstream analysis/baseline code that still reads legacy fields such as `raw_data`, score-aggregation config on the dataset object, and split/file grouping metadata in their current forms.

Those are **follow-on adoption tasks**, not hidden non-goals. Implementing this core is still worthwhile because it removes repeated loading, clarifies fit/apply semantics, centralizes alignment, and creates a clean public contract. But wiring the existing trainer/infer/analysis/CLI paths onto that new contract is a separate migration stream that must be planned explicitly, tested explicitly, and reviewed on its own merits.

A good implementation should therefore optimize first for a clean, well-tested standalone core API, not for preserving every legacy call site by accident.

---

## 2. Design decisions to preserve

### 2.1 Canonical data shape

Every loader and transform operates on a NumPy array with shape:

```text
(T, C, F)
T = time steps
C = channels
F = features per channel
```

A 1D or 2D source must be normalized by the loader (or a shared normalization helper) before it reaches transforms:

```text
(T,)    -> (T, 1, 1)
(T, C) -> (T, C, 1)
(T, C, F) -> unchanged
```

Do not allow individual transforms to silently use incompatible axis conventions.

### 2.2 One data container: `BaseDataset`

There is exactly **one** data-carrying class in this package: `BaseDataset`. There is no separate minimal container type (no `DataBlock`). Earlier drafts of this plan proposed a two-type split (`DataBlock` for pipeline-internal use, `BaseDataset` for pipeline output), but that separation only pays for itself when pre-split transforms need to perform complex, multi-axis-aware operations (for example, slicing an auxiliary quality-reference array in lockstep with a channel selection). This plan explicitly pushes that kind of complex, experiment-specific logic into custom loaders (§6.1, §7.2), so the extra type is not needed. One class, used consistently from load through final output, is simpler to learn, simpler to document, and matches how the existing `anldq` codebase already operates (a single mutable object built up stage by stage).

`BaseDataset` has two kinds of fields:

- **Data-contract fields** — read and written by loaders and transforms: `data`, `timestamps`, `labels`, `channel_labels`, `channel_names`, `feature_names`, `metadata`.
- **Orchestrator-managed fields** — set only by `PreprocessingPipeline`, never by a transform function: `split`, `data_tag`, `preprocess_steps`.

This is a documented convention enforced by code review and tests, not by a second class. See §5 for the full contract.

### 2.3 First-class aligned fields

Each `BaseDataset` instance has:

- `data`: `np.ndarray`, shape `(T, C, F)`;
- `timestamps`: `np.ndarray | None`, shape `(T,)` when present. Used for plotting and reporting downstream, so it is always carried through loading, filtering, and splitting whenever it is available;
- `labels`: `np.ndarray | None`, shape `(T,)` when present — one label per timestep (for example, "this timestep is anomalous"). Defaults to `None`; the pipeline never requires or synthesizes it, and only sets it when the source loader actually provides it;
- `channel_labels`: `np.ndarray | None`, shape `(T, C)` when present — one label per timestep *per channel* (for example, "this channel at this timestep is anomalous/bad"), for experiments that need finer-grained ground truth than a single per-timestep label. Also defaults to `None` and is never synthesized;
- `channel_names`: `list[str] | None`, length `C` when present;
- `feature_names`: `list[str] | None`, length `F` when present;
- `metadata`: `dict[str, Any]`, for units, source identifiers, fitted-transform state, provenance, and experiment-specific information.

The core must validate alignment after loading, after every transform, and after splitting:

```text
data.ndim == 3
data.shape[0] == len(timestamps)             # if timestamps is not None
data.shape[0] == labels.shape[0]             # if labels is not None
data.shape[0] == channel_labels.shape[0]     # if channel_labels is not None
data.shape[1] == channel_labels.shape[1]     # if channel_labels is not None
data.shape[1] == len(channel_names)          # if channel_names is not None
data.shape[2] == len(feature_names)          # if feature_names is not None
```

#### Auxiliary metadata alignment: one storage location, one operation

The current `anldq` codebase has four different, partially-inconsistent ways of keeping an auxiliary array in sync with the time axis: a declared `time_metadata_keys`/`channel_metadata_keys` list that is honored at some call sites but silently ignored at others; a `mask_group_lengths` function that specifically recomputes file/group boundary lengths from a keep-mask; GM2's bespoke rejoin of a quality table by timestamp *value* rather than row position; and `select_stable_channels`'s hardcoded, one-off channel-aligned slice. Four mechanisms exist because each was added later to solve one specific case, not because four cases are fundamentally different. This plan replaces all four with one:

**Rule:** any array placed under `metadata["aligned_time"][name]` must have `T` as its first dimension, and is sliced automatically, every time, by the same single function — `select_time_rows` (§6.2) — that already slices `data`/`timestamps`/`labels`/`channel_labels`. There is no second function, no declared-keys list to keep in sync, and no per-key special casing: `select_time_rows` iterates `metadata["aligned_time"]` and applies the same NumPy row-index/boolean-mask operation it applies to `data`, because `data[keep]` and `arbitrary_array[keep]` are the same operation. Anything *not* placed there is simply not managed by the core at all — the loader/transform that needs it is responsible for it, in the same way it is already responsible for every other ordinary `metadata` key.

That one rule covers every case the current four mechanisms exist for:

- **Per-group/file boundaries** (today's `file_lengths`): instead of storing a length-per-file list that a separate function has to recompute after every filter, store a plain row-aligned id array, e.g. `metadata["aligned_time"]["file_id"]` with shape `(T,)` (one integer per row, naming which file/group that row came from). `select_time_rows` slices it for free, with no dedicated group-length logic anywhere in the core; boundaries or per-group counts are recovered on demand with `np.unique(file_id, return_counts=True)` whenever something actually needs them.
- **Per-channel auxiliary arrays** (e.g. a reference signal read alongside `data`): stored with shape `(T, C, ...)` under the same `metadata["aligned_time"]` dict. `select_time_rows` slices its time axis exactly like it slices `data`'s time axis — no special case, because the operation is still "slice dimension 0 by `keep`." If a channel-selection transform *also* needs to narrow this array's channel axis, that transform does so the same way it narrows `data`'s channel axis — via `select_channels` (§6.2) — which is the pre-existing channel-slicing operation, not a second alignment mechanism invented for this purpose. Needing to narrow two different axes of two different arrays is not a reason to have two alignment *mechanisms*; it only means a transform that touches the channel axis calls the channel-slicing helper, the same one it already needs to slice `data` itself.
- **Value-rejoined tables that cannot be row-sliced at all** (GM2's case): if an array's rows don't correspond 1:1 with `data`'s rows (for example, a lookup table indexed by timestamp *value*, not row position), it must not be placed under `metadata["aligned_time"]` — doing so would be a validation error, since the core assumes anything stored there can be row-sliced. It is stored under an ordinary `metadata` key instead, and the custom loader/transform that produces and consumes it is responsible for re-joining it by value. This is not a third alignment mechanism; it is the explicit absence of one, for data that structurally cannot support row-slicing in the first place.

So there is one storage location (`metadata["aligned_time"]`) and one operation (`select_time_rows`'s row-slice, reused for every array placed there, including the `(T, C, ...)` case via the already-needed `select_channels` for the channel axis). The three bullets above are not three mechanisms to choose between — they are three *situations* that all resolve to either "put it under `metadata['aligned_time']` and it's handled automatically" or "don't, and handle it yourself," which is a binary choice, not a tiered one.

### 2.4 One full load and shared pre-split work

The public pipeline loads raw data **once**. It runs all pre-split transforms once, on the whole dataset, with no `apply`/`fit` scoping (§7.2). It then computes split indices once and makes train/val/test `BaseDataset` instances from the resulting full instance.

Post-split transforms run separately for each requested split, with `apply`/`fit` scoping available. This preserves correct semantics for train-fitted state while removing repeated file I/O and repeated shared preprocessing.

### 2.5 YAML describes scheduling; Python supplies operations

YAML selects:

- a loader name and loader parameters;
- ordered pre-split and post-split transforms;
- a split policy name and parameters.

Python supplies:

- loader functions;
- transform functions;
- split-policy functions;
- small registries mapping the YAML names to callables.

YAML must contain only short names, never dotted import paths or executable Python. Custom behavior is supplied exclusively by calling `register_loader`/`register_transform`/`register_split_policy` with real Python callables from experiment setup code before building a pipeline — this is a deliberate, explicit choice, not a placeholder for a future dotted-import-string feature. A separate, unrelated proposal in this repo (`FEAT_Dynamic_Imports.md`) describes a `target: package.module:function`-style dotted-string mechanism for the *existing* `anldq.datasets.registry` module; that proposal is explicitly out of scope for this plan and must not be adopted here. Pythonic, in-process registration avoids the security and fragility tradeoffs of string-based dynamic imports (arbitrary code execution from untrusted YAML, typo'd import paths failing at build time instead of config-load time, etc.) and keeps the registration call sites visible and reviewable as ordinary Python code.

### 2.6 Framework-owned rules

This package is intentionally a small public framework, not a loose collection of utilities. The framework owns the rules that must be consistent across downstream users:

- canonical data shape is always `(T, C, F)`;
- raw data is loaded exactly once per `build()` call;
- pre-split transforms always run once on the full dataset and never use `apply` or `fit`;
- split policies return indices only; the core materializes split datasets;
- post-split transforms run in deterministic `train -> val -> test` order;
- train-fitted state is scoped by configured step and must be serializable;
- row/channel alignment is performed only through core selection helpers;
- config validation rejects typos and unsafe combinations early;
- bundle metadata records provenance, final shapes, dropped rows, and fitted state.

Users configure these rules but do not redefine them. Plugin authors provide loaders, transforms, and split policies that obey these contracts.

---

## 3. Current implementation: what to retain and what to replace

The current `anldq.datasets` code is useful as a source of behavior but is not copied wholesale, and this new package does not import from it.

### 3.1 Useful existing behavior to port conceptually

- canonical `(T, C, F)` shape;
- timestamps and labels aligned to time rows;
- named YAML transforms with parameters;
- `apply: [train, val]` transform selection (post-split only, see §7.3);
- `fit: train` transform state shared into later splits;
- explicit split-policy functions;
- validation/error messages when a transform requires train-fitted state that does not exist.

### 3.2 Current behavior to replace

The current `build_data_splits(...)` in `src/anldq/datasets/builder.py` builds `train`, `val`, and `test` as separate `anldq.datasets.base.BaseDataset` instances. Each instance calls `BaseDataset._build()`, which invokes `_load_raw()` and executes the pipeline for that split independently. Verified in the current codebase: this means raw data loading, and any `after_load`/`pre_split`-equivalent transforms, run up to three times per `build_data_splits()` call. The only mitigation today is an opt-in, per-subclass, multi-file-only cache (`GM2Dataset`) or an always-on, never-evicted cache (`SPT3GDataset`) — both are workarounds for the same underlying architectural problem this plan solves directly. Because this plan's pipeline loads raw data exactly once per `build()` call by construction (§2.4), neither of these per-subclass caching mechanisms is needed in the new core at all — there is no dataset subclass hierarchy to attach a cache to, and no repeated load for a cache to avoid.

The new package replaces that with this single-pass orchestration, owned by a dedicated `PreprocessingPipeline` class (not `BaseDataset`):

```text
load full raw data once
-> normalize and validate
-> run pre_split transforms once, on the whole dataset
-> calculate train/val/test indices once
-> materialize three independent BaseDataset instances
-> run post_split transforms per split
-> return DatasetBundle
```

Also verified in the current codebase: production YAML configs use only `after_load` and `post_split` stages; `pre_split` is never used by any shipped config (it does appear in one unmerged feature-branch test, used interchangeably with `after_load` for the same transform — i.e., no real semantic distinction between the two stages has ever been established in practice). This plan collapses them into a single `pre_split` stage.

Note: two shipped SPT configs (`spt_baseline.yml`, `spt_seasonal.yml`) do configure `after_load` with `fit`/`apply` keys, via `select_stable_channels` (fit on train, reuse on val/test). This plan's `pre_split` stage deliberately does **not** support `fit`/`apply` (§7.2), so this is not a behavior-preserving collapse for that one transform — it is an intentional removal of that pattern. The replacement is not a pipeline-level fit/apply mechanism; it is moving the selection logic into the custom loader itself (§6.1), since the reference data it fits against is already available at load time, independent of the split. Every other shipped `after_load`/`pre_split`-equivalent transform in the current codebase runs the same way regardless of split and has no `fit`/`apply` dependency, so the collapse is behavior-preserving for all of them.

The new core uses exactly:

```yaml
loader:
pre_split:
post_split:
split:
```

If a future real use case needs a hook between built-in operations, add it only with an explicit ordering problem and test.

---

## 4. Target public API

The primary public API is intentionally small:

```python
from preprocessing_core import load_dataset_bundle

bundle = load_dataset_bundle("config.yaml")
```

Allow an optional registration function before loading a config when an experiment provides custom operations:

```python
from preprocessing_core import register_loader, register_transform, register_split_policy

register_loader("my_hdf5", load_my_hdf5)
register_transform("remove_bad_rows", remove_bad_rows)
register_split_policy("my_campaign_split", my_campaign_split)

bundle = load_dataset_bundle("config.yaml")
```

Also expose a lower-level free-function API for programmatic use and tests. `build_bundle(config)` is a thin wrapper around `PreprocessingPipeline` and is the one place the public API touches the orchestrator class directly:

```python
from preprocessing_core import load_config, build_bundle

config = load_config("config.yaml")
bundle = build_bundle(config)
```

```python
def build_bundle(config: PipelineConfig) -> DatasetBundle:
    return PreprocessingPipeline.from_config(config).build()


def load_dataset_bundle(path: str) -> DatasetBundle:
    return build_bundle(load_config(path))
```

Do not create `load_dataset()` in the new package; use the unambiguous `load_dataset_bundle()`. Do not expose `PreprocessingPipeline` or `BaseDataset` construction details as the primary entry point — most users only ever call `load_dataset_bundle()`.

---

## 5. Core data contract: `BaseDataset`

`BaseDataset` is the one data-carrying class in this package. An instance represents either the full (pre-split) dataset or one split (`train`/`val`/`test`). It is intentionally plain: a validated, attribute-complete container, not a `torch.utils.data.Dataset`, not a subclass of anything, and not the place pipeline orchestration logic lives.

This class is deliberately sized around what a dataset-shaped object needs to *carry data*, not around replicating every attribute the current `anldq.datasets.base.BaseDataset` exposes. It describes data only: arrays, timestamps, labels, and naming/metadata needed to interpret them. It is explicitly **not** responsible for evaluation concerns — score aggregation, channel-group-aware reduction, fitted-scaler lookup, or anything else that belongs to a training/inference result, not to the data itself. Those concerns are expected to live in downstream result types (e.g. a future `TrainResult`/`InferenceResult`, §18) built by whoever wires this bundle into training/inference, not on `BaseDataset`.

This plan's compatibility target is the new integrated `BaseDataset` + `DatasetBundle` contract defined here, not the full legacy surface of `anldq.datasets.base.BaseDataset`. The new `BaseDataset` intentionally exposes only the core dataset fields the rewritten library should rely on directly: `data`, `timestamps`, `labels`, `channel_labels`, `channel_names`, `feature_names`, `metadata`, plus orchestrator-owned `split`, `data_tag`, and `preprocess_steps`. Convenience properties such as `.shape`, `.n_channels`, `.n_features`, and `__len__()` are derived from `data` rather than stored separately. `timestamps`, `labels`, and `channel_labels` remain first-class top-level fields — they are not moved into `metadata` — because they are part of the alignment contract and, in the case of `timestamps`, are required by built-in split policies such as the seasonal split (§6.3.2). Legacy analysis- or baseline-specific concerns such as score aggregation, channel groups, or the old tuple-returning CLI contract are explicit downstream or follow-on migration work, not fields this plan preserves on `BaseDataset` itself:

```python
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class BaseDataset:
    """One full or split dataset in canonical (T, C, F) form."""

    data: np.ndarray
    timestamps: np.ndarray | None = None
    labels: np.ndarray | None = None               # (T,) per-timestep label, optional
    channel_labels: np.ndarray | None = None        # (T, C) per-timestep-per-channel label, optional
    channel_names: list[str] | None = None
    feature_names: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # set only by PreprocessingPipeline; transforms must not modify these
    split: str | None = None
    data_tag: str | None = None
    preprocess_steps: list[str] = field(default_factory=list)

    def validate(self) -> None:
        ...

    def copy(self) -> "BaseDataset":
        ...

    def with_split(self, split: str, data_tag: str | None = None) -> "BaseDataset":
        """Return a copy with `split`/`data_tag` set. Called only by PreprocessingPipeline."""
        result = self.copy()
        result.split = split
        result.data_tag = data_tag
        return result

    def __len__(self) -> int:
        return self.data.shape[0]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def n_features(self) -> int:
        return self.data.shape[2]
```

Notes on fields that the current `anldq.datasets.base.BaseDataset` has but this one deliberately omits, and why each is safe to drop:

- **`.raw_data`** (pre-scaling array, separate from `.data`): not a required `BaseDataset` field. The new contract derives `.n_channels` from `data.shape[1]` directly, so no second array is needed for core behavior. If a loader or downstream workflow wants to retain an unprocessed copy for debugging, benchmarking, or later analysis, it may do so **optionally** under `metadata["raw_data"]`; the default v1 behavior does not retain raw data automatically, to avoid doubling memory usage.
- **`.fitted_scaler` / `.fitted_baseline_state`** as dataset attributes: not needed, and deliberately not carried on any split's dataset object at all. Fitted preprocessing state lives exclusively at the bundle level, in `DatasetBundle.metadata["fitted_state"]` (§7.4). This is a simplification relative to today's design, where the literal fitted object is threaded through dataset-to-dataset constructor arguments; under this plan, nothing reads a fitted scaler/baseline off `bundle.train`/`bundle.val`/`bundle.test` — code that needs it (for example, inference-time rescaling) reads `bundle.metadata["fitted_state"]` instead.
- **`.channel_groups` / group-level score aggregation**: not a `BaseDataset` field. Group-aware evaluation/aggregation is downstream, experiment-specific analysis/result logic (e.g. a future `InferenceResult`, §18); it is expected to be built in custom downstream evaluation code. When a loader or transform does know channel-group identity (for example, detector subsystem groupings), the supported convention is to carry it as an ordinary `metadata["channel_groups"]` array (length `C`, aligned the same way `channel_names` is), not as a dedicated `BaseDataset` field — downstream result types read it from there if they need group-aware reduction.
- **A `GM2Dataset`-style class-name check**: not applicable, since there is no subclassing in this design. Any behavior that today keys off `ds.__class__.__name__` must instead key off an explicit flag the loader sets in `metadata` (for example `metadata["preserve_feature_axis"] = True`), which is both more explicit and robust to renaming.
- **`enable_load_cache` / `clear_load_cache` classmethods**: not needed. These exist today purely to work around repeated raw loads across train/val/test construction; this plan's pipeline loads raw data exactly once per `build()` call by construction (§2.4), so there is nothing left for a cache to avoid.

Required invariants (checked by `validate()`):

1. `data` is a NumPy array with exactly three dimensions.
2. Its time dimension is non-empty; v1 rejects empty datasets/splits by default (see §9).
3. `timestamps`, when supplied, is 1D and length `T`.
4. `labels`, when supplied, is 1D and length `T`. Labels are optional; many experiments have none, and the pipeline must work identically whether or not labels are present.
5. `channel_labels`, when supplied, has shape `(T, C)`. Also optional and never synthesized.
6. `channel_names`, when supplied, has length `C`.
7. `feature_names`, when supplied, has length `F`.
8. Any array under `metadata["aligned_time"]`, when present, has first dimension `T` (§2.3).

Rules:

- Each split must receive its own `BaseDataset` instance and its own `metadata` dict copy; splits must not share mutable arrays/metadata unexpectedly.
- `copy()` must deep-copy `data`/`timestamps`/`labels`/`channel_labels` (or at minimum guarantee no aliasing across splits) and deep-copy `metadata` into a new dict (including any arrays nested under `metadata["aligned_time"]`), so no split can mutate another split's fitted-state or auxiliary arrays by accident.
- Transform functions may only read/write the data-contract fields (`data`, `timestamps`, `labels`, `channel_labels`, `channel_names`, `feature_names`, `metadata`). `split`, `data_tag`, and `preprocess_steps` are set exclusively by `PreprocessingPipeline`. This is documented and covered by a test that a transform cannot silently corrupt orchestrator bookkeeping.
- Fields are ordinary mutable dataclass attributes (the dataclass is not frozen), so code that needs to subsample or reassign `.data`/`.labels` in place (as the current codebase's opt-in sweep tooling does) continues to work; the documented contract is simply that *pipeline transforms* do not do this, not that the object is immutable.
- This `BaseDataset` is defined in `preprocessing_core.datasets` and is a standalone class, independent of and not related by inheritance to `anldq.datasets.base.BaseDataset`. Reaching feature parity with the existing `anldq` trainer/infer CLI (`src/anldq/trainer`, `src/anldq/infer`, `src/anldq/cli`) is **not** a goal of this plan; that wiring, including any call-site changes those modules would need (e.g. deriving `.n_channels` from `.data` instead of `.raw_data`, or replacing the `GM2Dataset` class-name check with an explicit `metadata` flag), is explicit, separate, future migration work (§18). This plan's scope ends at producing a valid `DatasetBundle`; evaluation logic (score aggregation, channel-group-aware reduction, and the wider, more experiment-specific attribute surface used by `anldq`'s analysis/plotting code) is explicitly left to downstream result types and custom evaluation code, not this plan.

### 5.1 `DatasetBundle`

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class DatasetBundle:
    train: BaseDataset
    val: BaseDataset
    test: BaseDataset
    metadata: dict[str, Any] = field(default_factory=dict)
```

Bundle metadata should contain information shared across the three splits. `DatasetBundle.metadata["fitted_state"]` is the canonical storage location for all fitted post-split transform state; split datasets themselves do not carry duplicated fitted-state attributes. Bundle metadata may also contain small split-policy provenance such as dropped-row counts. For example:

```python
{
    "config_path": "configs/example.yaml",
    "config_version": 1,
    "split_policy": "chronological",
    "split_parameters": {"ratios": [0.6, 0.2, 0.2]},
    "canonical_shape": "(T, C, F)",
    "pre_split_steps": ["fill_nan"],
    "post_split_steps": {
        "train": ["drop_invalid_timesteps", "standard_scale"],
        "val": ["drop_invalid_timesteps", "standard_scale"],
        "test": ["standard_scale"],
    },
    "dropped_rows": {"gap_train_val": 5, "gap_val_test": 5},
    "fitted_state": {
        "post_split:1:standard_scale": {"mean": [...], "std": [...]},
    },
}
```

Do not duplicate large per-split arrays in the bundle metadata. Fitted transform state (§8) is the one exception that is intentionally duplicated into bundle metadata, since it is small and needed for reproducibility/inference-time reuse.

---

## 6. Loader, transform, and split-policy contracts

### 6.1 Loader contract

A loader is a normal Python callable registered under a short YAML name:

```python
def load_csv_timeseries(*, path: str, timestamp_column: str | None = None) -> BaseDataset:
    ...
```

Requirements:

- return a valid `BaseDataset` (or a documented raw result immediately converted by one central helper);
- normalize to `(T, C, F)` before return;
- provide timestamps/labels/channel_labels/channel_names/feature_names only when actually available;
- not split data;
- not apply configurable preprocessing;
- not depend on private paths, packages, or global state.

**Custom, experiment-specific selection logic (any non-generic, dataset-specific filtering that depends on private data formats, private reference files, or private domain knowledge) belongs in a custom loader, registered via `register_loader`, not in a pre-split transform.** This is a deliberate scope decision: it keeps the public pre-split transform set small and fully generic, and keeps genuinely experiment-specific logic entirely in private plugin code that only needs to produce a valid `BaseDataset`. This plan deliberately does **not** ship anything resembling the current codebase's `select_stable_channels` (fitting a channel-selection mask against an externally supplied reference array, `src/anldq/datasets/process/filters.py`) as a public pre-split or post-split transform. That logic is considered unwanted behavior going forward — not a capability to preserve as-is — and it does not need a `fit`/`apply` pipeline stage to be reproduced: the reference array it fits against (e.g. SPT's `quality_reference_response`) is itself loaded independently of train/val/test split assignment (`src/anldq/datasets/data_loaders/spt.py`), so computing the stable-channel mask and applying it belongs entirely inside the custom loader, before any `BaseDataset` is returned and before splitting happens. A custom loader for SPT-style data is expected to perform this channel selection internally and simply return an already-filtered `BaseDataset`; the public pre-split stage is not the place for it, and no pipeline-level `fit`/`apply` support is needed to replace it.

Custom loaders may be complex, but they should still be decomposed. A loader should orchestrate source-specific assembly; it should not become an unstructured replacement for the old dataset subclasses. For complex experiments, split private loader code into focused helpers such as readers, normalizers, quality filters, channel selectors, and metadata assemblers. The public core cannot enforce this inside private plugins, but the extension docs should recommend it strongly.

The core ships a small set of public generic loaders in v1:

- `npy`: `.npy` array with optional sidecar timestamps/labels/channel_labels;
- `npz`: named `data`, optional `timestamps`, `labels`, `channel_labels`;
- `csv`: via pandas, with documented value/timestamp/label column configuration (no `channel_labels` support, since a 2D CSV has no natural `(T, C)` label layout — experiments needing `channel_labels` from CSV should use a custom loader);
- `hdf5`: a minimal `h5py`-based loader reading named datasets by key (data/timestamps/labels/channel_labels), mirroring the same `dataset[:]`-into-memory pattern already used everywhere `h5py` appears in this repository today — there is no chunked/streaming/complex-group-hierarchy support, and none is needed for a generic loader.

Parquet, remote stores, and experiment-specific formats are not included in v1. They are expected to be supplied by private plugin packages via `register_loader`.

### 6.2 Transform contract

Use one transform signature for both stages:

```python
def transform_name(
    dataset: BaseDataset,
    *,
    state: dict[str, object],
    fit: bool,
    **params: object,
) -> BaseDataset:
    ...
```

Rules:

- transforms accept and return a `BaseDataset`, not a raw array; this prevents accidental timestamp/label misalignment;
- transforms must not read or write `split`, `data_tag`, or `preprocess_steps`;
- transforms may edit in place internally, but returning a (possibly new) `BaseDataset` is the documented contract; choose one convention (return a new object) and apply it consistently;
- transforms must preserve `(T, C, F)` unless their documented purpose is channel/feature selection;
- time filtering must apply the same row selection to `data`, `timestamps`, `labels`, `channel_labels`, and every array under `metadata["aligned_time"]` (§2.3) — this is exactly what `select_time_rows` below does, so transforms should call it rather than slicing `data` directly;
- channel selection must update `channel_names` and the channel axis of `channel_labels` accordingly; feature selection must update `feature_names` accordingly;
- `fit=True` means the transform is being fit on its configured fitting split;
- fitted state must be **plain, serializable data only** — NumPy arrays, lists, scalars, or dicts thereof. References to sklearn estimator objects or any other non-serializable object are rejected. This is a hard requirement, not a soft preference: the current codebase passes raw sklearn scaler instances by reference between split builds, which this plan explicitly forbids going forward.

Provide central helpers rather than asking each transform to repeat alignment logic:

```python
select_time_rows(dataset: BaseDataset, keep_or_indices) -> BaseDataset
select_channels(dataset: BaseDataset, keep_or_indices) -> BaseDataset
```

`select_time_rows` applies one row selection to `data`, `timestamps`, `labels`, `channel_labels`, and every array under `metadata["aligned_time"]` in one place, so no transform, loader, or split policy ever reimplements this slicing by hand. `select_channels` does the equivalent for the channel axis: `data`, `channel_names`, and the channel axis of `channel_labels`. Both helpers are required in v1 (not deferred) since the shipped `keep_channels`/`drop_channels` pre-split transforms (§12) need `select_channels`, and every post-split row filter needs `select_time_rows`.

### 6.3 Split-policy contract

A split policy receives the fully preprocessed full `BaseDataset` and returns indices, not copied datasets:

```python
def chronological_split(
    dataset: BaseDataset,
    *,
    ratios: tuple[float, float, float],
    gap: int = 0,
) -> dict[str, np.ndarray]:
    """Return ordered row indices for train, val, test."""
```

Required result:

```python
{
    "train": train_indices,
    "val": val_indices,
    "test": test_indices,
}
```

All split policies in v1 are implemented with plain NumPy indexing/slicing. No machine-learning library is required: a review of every split strategy in the current `anldq` codebase (`RatioSplit`, `FileWiseRatioSplit`, `YearWiseRatioSplit`, `SeasonStratifiedBlockSplit`, `RandomFileWiseRatioSplit`) confirms all of them reduce to NumPy index arithmetic plus, in randomized cases, a seeded NumPy random generator for shuffling. scikit-learn's split utilities (e.g. `train_test_split`) are not used by any current policy and are not a good fit here regardless, since they are built for two-way splits and would need to be chained awkwardly to produce three. **Do not add a scikit-learn dependency for splitting.** Use `np.random.default_rng(seed)` for any policy that needs randomness; do not use Python's built-in `hash()` for seeding, which the current codebase does in one place and which relies on undocumented, implementation-specific integer-hash behavior for reproducibility.

### 6.3.1 Leakage controls

Because adjacent rows in time-series data are often correlated, built-in split policies support an explicit, optional gap. v1 keeps the split-policy return contract intentionally simple — split policies still return only plain split-index dicts — and any small dropped-row provenance needed for bundle metadata is computed by the built-in split implementation or a small pipeline-owned helper, not by a richer `SplitResult` abstraction.

- `gap` defaults to `0` (no dropped rows), preserving the simplest possible behavior unless a user opts in.
- For `chronological`, `gap` is a row count dropped at the train/val and val/test boundaries.
- For `seasonal_block`, `gap` is a row count trimmed from both ends of each selected block within each `(year, season)` stratum.
- Built-in split policies should record simple dropped-row counts in bundle metadata (e.g. `metadata["dropped_rows"]`). Exact dropped-index provenance is not required in v1.

Validation requirements:

- all three names (`train`, `val`, `test`) are present;
- each index array is 1D integer positions;
- indices are within `[0, T)`;
- no overlap between any two splits — this is a hard rejection, not a warning;
- all selected rows within a split are unique;
- **all splits are non-empty — this is a hard rejection.** A policy or configuration that would produce an empty split (for example, extreme ratios combined with very small `n`, or a `gap` large enough to consume an entire split) must raise a clear `ConfigError` rather than silently returning a zero-length split. This closes a confirmed bug in the current codebase, where several split classes can silently produce an empty `val` or `test` split for small `n` with no validation anywhere.
- policies that claim chronological behavior return ordered indices.

The builder uses `select_time_rows` to turn these indices into three independent `BaseDataset` instances.

### 6.3.2 Built-in split policies in v1

v1 ships exactly three built-in split policies:

- `chronological`: contiguous train/val/test split over row order;
- `index_ranges`: explicit half-open row ranges per split;
- `seasonal_block`: year/season-stratified assignment of contiguous time blocks.

`seasonal_block` is included in v1 because it has a minimal, generic contract: it requires only `dataset.timestamps`, not experiment-specific metadata. It derives calendar year/month from `timestamps`, maps months into configured seasons, divides each `(year, season)` stratum into contiguous blocks, assigns whole blocks to train/val/test using seeded NumPy shuffling, and optionally trims a block-edge `gap` within each selected block. If `timestamps` are missing or cannot be converted to calendar datetimes, `seasonal_block` raises a configuration/runtime error immediately.

---

## 7. YAML configuration design

### 7.1 Required top-level schema

Use this schema in every config:

```yaml
version: 1

loader:
  type: npz
  params:
    path: ./data/example.npz

pre_split:
  - name: fill_nan
    params:
      value: 0.0

split:
  policy: chronological
  params:
    ratios: [0.6, 0.2, 0.2]
    gap: 0

post_split:
  - name: drop_invalid_timesteps
    apply: [train, val]
    params:
      require_finite: true

  - name: standard_scale
    fit: train
    apply: [train, val, test]
    params:
      mode: per_channel_feature

metadata:
  dataset_name: example
```

Top-level keys:

| Key | Required | Purpose |
| --- | --- | --- |
| `version` | yes | Schema version; v1 is `1`. |
| `loader` | yes | Select loader and its parameters. |
| `pre_split` | no | Ordered transforms on the full dataset before splitting. |
| `split` | yes | Split policy and its parameters. |
| `post_split` | no | Ordered transforms after split creation. |
| `metadata` | no | User-facing bundle-level metadata. |

Reject unknown top-level keys in v1. Strict validation prevents typo-driven silent behavior — a confirmed gap in the current codebase, where a misspelled field (e.g. `apl:` instead of `apply:`) silently falls back to a default instead of raising.

#### Framework defaults

Unless explicitly configured:

- `pre_split` is an empty list;
- `post_split` is an empty list;
- post-split `apply` defaults to all splits: `train`, `val`, `test`;
- `fit` is absent by default;
- split `gap` defaults to `0` for built-in policies;
- `metadata` defaults to `{}`;
- `state_key` defaults to `{stage}:{position}:{name}`;
- labels and channel labels default to `None` when the loader does not provide them.

Defaults must be applied in config parsing, not scattered throughout transforms.

### 7.2 Pre-split transforms

`pre_split` is an ordered list. Every transform runs once on the full dataset before split indices are generated, and **always applies to the entire dataset — there is no `apply` or `fit` concept at this stage.** This is a simplification from earlier drafts of this plan: pre-split transforms are for generic, dataset-wide, order-independent operations only (deterministic channel/feature pruning, NaN filling, timestamp validation). Any transform that needs to behave differently per split, or that needs to fit state on one split and reuse it elsewhere, belongs in `post_split`. Any transform that needs complex, experiment-specific logic (for example, selecting channels based on an externally supplied quality reference) belongs in a custom loader (§6.1), not here.

```yaml
pre_split:
  - name: validate_monotonic_timestamps
  - name: interpolate_nan_per_channel
    params:
      require_all_filled: false
  - name: keep_channels
    params:
      names: [sensor_a, sensor_b]
```

Pre-split transforms must not have `apply` or `fit` keys in their YAML entries; the config loader rejects a `pre_split` entry that includes either key.

Transforms shipped in the public core for this stage (see §12 for the full list):

- `fill_nan`
- `interpolate_nan_per_channel`
- `keep_channels` / `drop_channels`
- `keep_features` / `drop_features`
- `validate_monotonic_timestamps`

### 7.3 Post-split transforms

`post_split` is an ordered list. The pipeline processes a transform across splits in fixed order:

```text
for each configured transform:
    train, then val, then test
```

This deterministic ordering lets `fit: train` create state before val/test reuse it.

Basic transform:

```yaml
post_split:
  - name: drop_invalid_timesteps
    apply: [train, val]
    params:
      require_positive: true
      min_valid_fraction: 0.95
```

Train-fitted transform (scaling is implemented exactly this way — there is no separate "scaling stage" in the pipeline, it is an ordinary post-split transform with `fit`/`apply`):

```yaml
post_split:
  - name: standard_scale
    fit: train
    apply: [train, val, test]
    params:
      mode: per_channel_feature
```

Meaning:

1. Run on `train` with `fit=True` and save state.
2. Run on `val` with `fit=False`, reusing train state.
3. Run on `test` with `fit=False`, reusing train state.

Optional explicit state key:

```yaml
  - name: standard_scale
    state_key: standard_scaler_v1
    fit: train
    apply: [train, val, test]
```

Without `state_key`, derive a predictable key from the transform list position and name, e.g. `post_split:0:standard_scale`. This avoids accidental sharing between two instances of the same transform.

Rules for post-split fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | Registered transform name. |
| `params` | no | Keyword parameters passed to the transform. |
| `apply` | no | Split names; default is all splits. |
| `fit` | no | Must be `train` if present; any other value is rejected at config-load time. Must also be included in `apply`. |
| `state_key` | no | Stable explicit name for reusable state. |

Validation rules:

- `apply` must be a non-empty subset of `train`, `val`, `test`;
- `fit`, when supplied, **must be `"train"`**. Any other value (`"val"`, `"test"`) is rejected with a clear configuration error at load time. This closes a gap in the current codebase, which accepts any `fit` value and only fails later, confusingly, when state turns out to be missing;
- `fit` must also be in `apply`; a transform with `fit: train` and `apply: [val]` is invalid because train never runs and never produces state;
- the fixed `train -> val -> test` execution order means val/test always see train's state already populated when they run;
- if a transform needs fit state but it is absent, raise a clear configuration error naming the transform, the state key, and the split that failed.

### 7.4 Fitted state persistence

Fitted post-split transform state (for example, a `standard_scale` transform's per-channel mean/std) must not be a throwaway in-memory artifact. It is:

1. recorded in `DatasetBundle.metadata["fitted_state"][state_key]`, so it travels with the bundle and is visible to any code inspecting bundle metadata;
2. optionally persisted to the output/run directory as a small, human-inspectable file (for example, `fitted_state.json` or `.npz` next to the pipeline's other output artifacts), so the exact fitted preprocessing used for one run can later be reapplied to new, unseen data (for example, during online/production inference) without rerunning the fitting step. The exact persistence format and the API for reloading/reapplying saved state against new raw data is implementation detail to resolve during Phase 6 (§15); the requirement here is only that fitted state is a first-class, inspectable, persistable artifact, not something silently discarded when `build()` returns.

### 7.5 Example complete YAML

```yaml
version: 1

loader:
  type: npz
  params:
    path: ./data/public_example.npz
    data_key: data
    timestamps_key: timestamps
    labels_key: labels            # optional; omit if the source has no per-timestep labels
    # channel_labels_key: channel_labels   # optional; (T, C) per-channel labels, omitted here

pre_split:
  - name: validate_monotonic_timestamps
  - name: interpolate_nan_per_channel
    params:
      require_all_filled: false

split:
  policy: chronological
  params:
    ratios: [0.6, 0.2, 0.2]
    gap: 5

post_split:
  - name: filter_quantile_timesteps
    apply: [train, val]
    params:
      require_positive: true
      low_quantile: 1.0
      high_quantile: 99.0
      min_valid_fraction: 0.95

  - name: standard_scale
    state_key: standard_scaler
    fit: train
    apply: [train, val, test]
    params:
      mode: per_channel_feature

metadata:
  dataset_name: public-example
  description: Example only; replace loader path locally.
```

This expresses the intended order:

```text
load once
-> validate timestamps and interpolate once before split
-> chronological split, with a 5-row gap dropped at each boundary
-> filter train and val only
-> fit scaler on train and apply that fitted scaler to train/val/test
-> DatasetBundle
```

---

## 8. Pipeline algorithm

Implement one orchestration class, `PreprocessingPipeline`, in `pipeline.py`. It should be short enough to understand from top to bottom. `build_bundle(config)` (§4) is a one-line free-function wrapper around it and is the only sanctioned public entry point other than `load_dataset_bundle()`; there is no separate, independent implementation of the algorithm duplicated as a bare function.

### 8.1 `PreprocessingPipeline` orchestration

`PreprocessingPipeline` is a thin orchestrator: it is not a PyTorch dataset, not a data container, and not the place where transform logic lives. It holds a parsed `PipelineConfig`, resolves YAML names into registered Python callables, builds an ordered schedule of loader/pre-split/split/post-split work, runs that schedule, and returns a `DatasetBundle`. All data lives in `BaseDataset` and `DatasetBundle`. Loader, transform, split, validation, and selection behavior stays in lower-level functions.

```text
YAML config
-> load_config(path)
-> PipelineConfig
-> PreprocessingPipeline.from_config(config)
-> resolve registered loader / transforms / split policy
-> schedule steps in YAML order
-> build()
-> DatasetBundle(train, val, test, metadata)
```

A readable `PreprocessingPipeline` should look roughly like this:

```python
class PreprocessingPipeline:
    """Thin orchestrator for one configured preprocessing pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.loader = None
        self.pre_split_steps: list[ScheduledStep] = []
        self.split_policy = None
        self.post_split_steps: list[ScheduledStep] = []
        self.state: dict[str, dict[str, object]] = {}

    @classmethod
    def from_yaml(cls, path: str) -> "PreprocessingPipeline":
        config = load_config(path)
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "PreprocessingPipeline":
        pipeline = cls(config)
        pipeline.resolve()
        return pipeline

    def resolve(self) -> None:
        """Resolve YAML-selected names to registered callables and schedules."""
        self._resolve_loader()
        self._resolve_pre_split_steps()
        self._resolve_split_policy()
        self._resolve_post_split_steps()

    def build(self) -> DatasetBundle:
        """Run the complete one-load pipeline."""
        full = self._load_dataset()
        full = self._run_pre_split(full)

        splits = self._split_dataset(full)
        splits = self._run_post_split(splits)

        return self._make_output(splits)
```

The top-level methods deliberately read like the pipeline diagram. All details are delegated.

#### Delegation boundaries

`PreprocessingPipeline` sequences work and owns build-time state. It must delegate details as follows:

| Responsibility | Owner |
| --- | --- |
| YAML parsing and schema validation | `config.py` |
| callable lookup and duplicate registration rules | `registry.py` |
| transform scheduling and `apply`/`fit` validation | small helpers in `pipeline.py` or `validation.py` |
| row/channel slicing and alignment | `selection.py` |
| dataset and split-index validation | `validation.py` / `BaseDataset.validate()` |
| loader-specific file parsing | `loaders/*` or external plugin loaders |
| transform-specific data changes | `transforms/*` or external plugin transforms |
| split index calculation | `splits/*` or external plugin policies |
| bundle metadata assembly | `make_bundle_metadata()` helper, not inline pipeline code |
| fitted-state serialization checks | validation helper, used by pipeline before metadata/persistence |
| logging policy | pipeline logs stage boundaries; helpers do not spam logs |

If a method on `PreprocessingPipeline` grows beyond sequencing and context wrapping, extract the decision into a helper function.

#### Loader block

```python
    def _resolve_loader(self) -> None:
        self.loader = get_loader(self.config.loader.type)

    def _load_dataset(self) -> BaseDataset:
        dataset = self.loader(**self.config.loader.params)
        dataset = normalize_to_base_dataset(dataset)
        dataset.validate()
        return dataset
```

The loader block must run exactly once per `build()` call.

#### Pre-split block

```python
    def _resolve_pre_split_steps(self) -> None:
        self.pre_split_steps = schedule_transform_steps(
            stage="pre_split",
            step_configs=self.config.pre_split,
            allow_apply=False,
            allow_fit=False,
        )

    def _run_pre_split(self, dataset: BaseDataset) -> BaseDataset:
        for step in self.pre_split_steps:
            dataset = run_transform_step(
                step=step,
                dataset=dataset,
                state={},
                fit=False,
            )
            dataset.validate()
        return dataset
```

Pre-split transforms always run once on the full dataset. They do not use `apply` or `fit` in v1.

#### Split block

```python
    def _resolve_split_policy(self) -> None:
        self.split_policy = get_split_policy(self.config.split.policy)

    def _split_dataset(self, full: BaseDataset) -> dict[str, BaseDataset]:
        indices = self.split_policy(full, **self.config.split.params)
        validate_split_indices(indices, n_rows=full.data.shape[0])

        # data_tag is the config's metadata.dataset_name, unchanged across all three
        # splits — it identifies which source dataset a split came from, not which split it is.
        data_tag = self.config.metadata.get("dataset_name")

        return {
            "train": select_time_rows(full, indices["train"]).with_split("train", data_tag),
            "val": select_time_rows(full, indices["val"]).with_split("val", data_tag),
            "test": select_time_rows(full, indices["test"]).with_split("test", data_tag),
        }
```

Split policies return indices only. The core owns selection and alignment through `select_time_rows`, and owns setting `split`/`data_tag` on the resulting `BaseDataset` instances via `with_split()` (§5) — transforms never set these fields themselves. `data_tag` is read once from `config.metadata["dataset_name"]` (falling back to `None` if absent) and applied identically to all three splits, mirroring how the current `anldq` codebase's `data_tag` identifies the source dataset, not the split.

#### Post-split block

```python
    def _resolve_post_split_steps(self) -> None:
        self.post_split_steps = schedule_transform_steps(
            stage="post_split",
            step_configs=self.config.post_split,
            allow_apply=True,
            allow_fit=True,
        )

    def _run_post_split(
        self,
        splits: dict[str, BaseDataset],
    ) -> dict[str, BaseDataset]:
        for step in self.post_split_steps:
            step_state = self.state.setdefault(step.state_key, {})

            for split_name in ("train", "val", "test"):
                if split_name not in step.apply:
                    continue

                splits[split_name] = run_transform_step(
                    step=step,
                    dataset=splits[split_name],
                    state=step_state,
                    fit=step.fit == split_name,
                    split_name=split_name,
                )
                splits[split_name].validate()

        return splits
```

Post-split steps are scheduled exactly as written in YAML. Each step can choose which splits it applies to with `apply`. Train-fitted state is scoped by `state_key`, persisted into bundle metadata (§7.4), and reused by later splits in the same `build()` call.

#### Output block

```python
    def _make_output(
        self,
        splits: dict[str, BaseDataset],
    ) -> DatasetBundle:
        return DatasetBundle(
            train=splits["train"],
            val=splits["val"],
            test=splits["test"],
            metadata=make_bundle_metadata(
                config=self.config,
                pre_split_steps=self.pre_split_steps,
                post_split_steps=self.post_split_steps,
                splits=splits,
                fitted_state=self.state,
            ),
        )
```

#### Dynamic scheduler

The dynamic scheduler is just a converter from parsed YAML step configs to executable `ScheduledStep` objects. It should be small enough to audit easily.

```python
@dataclass
class ScheduledStep:
    stage: str
    position: int
    name: str
    fn: Callable[..., BaseDataset]
    params: dict[str, object]
    apply: tuple[str, ...]
    fit: str | None
    state_key: str
```

```python
def schedule_transform_steps(
    *,
    stage: str,
    step_configs: list[TransformConfig],
    allow_apply: bool,
    allow_fit: bool,
) -> list[ScheduledStep]:
    scheduled = []

    for position, step_config in enumerate(step_configs):
        name = step_config.name
        fn = get_transform(name)
        params = step_config.params or {}

        if not allow_apply and step_config.apply is not None:
            raise ConfigError(f"{stage} step {name!r} cannot use apply")

        if not allow_fit and step_config.fit is not None:
            raise ConfigError(f"{stage} step {name!r} cannot use fit")

        apply = tuple(step_config.apply or ("train", "val", "test"))
        fit = step_config.fit
        validate_apply_and_fit(stage=stage, name=name, apply=apply, fit=fit)

        state_key = step_config.state_key or f"{stage}:{position}:{name}"

        scheduled.append(
            ScheduledStep(
                stage=stage,
                position=position,
                name=name,
                fn=fn,
                params=params,
                apply=apply,
                fit=fit,
                state_key=state_key,
            )
        )

    return scheduled
```

Running a scheduled transform is also centralized, and includes full error context (stage, transform name, and split name, when applicable):

```python
def run_transform_step(
    *,
    step: ScheduledStep,
    dataset: BaseDataset,
    state: dict[str, object],
    fit: bool,
    split_name: str | None = None,
) -> BaseDataset:
    context = f"{step.stage} transform {step.name!r}"
    if split_name is not None:
        context += f" for split {split_name!r}"

    try:
        result = step.fn(dataset, state=state, fit=fit, **step.params)
    except Exception as exc:
        raise PipelineError(f"{context} failed") from exc

    if not isinstance(result, BaseDataset):
        raise PipelineError(f"{context} did not return BaseDataset")

    return result
```

YAML remains declarative. It never imports Python. Names resolve through explicit registration:

```python
register_loader("npz", load_npz)
register_transform("fill_nan", fill_nan)
register_transform("standard_scale", standard_scale)
register_split_policy("chronological", chronological_split)
```

Experiment code can register custom behavior before building:

```python
register_transform("remove_detector_outages", remove_detector_outages)
bundle = load_dataset_bundle("config.yaml")
```

The core should not care whether a registered function is built in or user-provided.

### 8.2 Important implementation details

1. **Load only once.** The loader must be called exactly one time per `build()` invocation. Add a unit test that proves this.
2. **Do not mutate `full` after split creation.** Split datasets should own their arrays after selection. Slicing can be a view in NumPy, so use `.copy()` in `select_time_rows` unless immutability is rigorously enforced. V1 should favor safety over saving this memory.
3. **Pre-split transforms use no train fit state.** This avoids accidental future leakage. Train-fitted behavior belongs in `post_split`.
4. **State is scoped per configured step.** Do not share one global transform state across unrelated steps.
5. **Fitted state is a first-class artifact, not a throwaway.** It is recorded in bundle metadata and may be persisted to disk (§7.4), not merely held in an internal dict that vanishes once `build()` returns.
6. **Post-split filtering is allowed to change split lengths.** It may leave split lengths different from the original ratios; this is expected and recorded in bundle metadata.
7. **Record provenance.** Store configured step names, final shapes/lengths, and dropped-row counts from split gaps in bundle metadata. Do not store data values or private source paths unless the caller explicitly asks.
8. **Transforms must not write to `split`, `data_tag`, or `preprocess_steps`.** These are orchestrator-owned fields on `BaseDataset`; a test should assert a misbehaving transform cannot corrupt them.

---

## 9. Split policies for v1

v1 ships exactly three split policies, all implemented with plain NumPy indexing — no external library is required for any split policy, now or in the set of policies the current `anldq` codebase already implements (confirmed: every existing policy, including randomized ones, reduces to NumPy index arithmetic and, at most, a seeded random generator for shuffling). The built-ins are `chronological`, `index_ranges`, and `seasonal_block`. More sophisticated, experiment-specific split policies still belong in private plugin code registered via `register_split_policy`, but v1 includes `seasonal_block` directly because it depends only on generic timestamp semantics, not proprietary metadata or detector-specific logic.

### 9.1 `chronological`

```yaml
split:
  policy: chronological
  params:
    ratios: [0.6, 0.2, 0.2]
    gap: 0
```

- Split contiguous ordered time ranges using plain NumPy slicing.
- No shuffle.
- Best default for time-series experiments.
- Validate ratios are positive and sum to 1 within a documented tolerance.
- Define one rounding rule and test it: allocate train and val with floor, give remainder to test.
- Optional `gap` (row count, default `0`) drops rows at each split boundary to reduce autocorrelation leakage (§6.3.1).
- Reject any configuration that would leave a split empty, including as a result of `gap` consuming an entire split.

### 9.2 `index_ranges`

```yaml
split:
  policy: index_ranges
  params:
    train: [0, 600]
    val: [600, 800]
    test: [800, 1000]
    gap: 0
```

- Use half-open `[start, stop)` intervals, implemented as plain NumPy slicing.
- Useful for reproducible externally decided boundaries, and as the simplest possible policy for exercising/validating the pipeline architecture.
- Validate coverage/non-overlap and bounds.
- Optional `gap` trims rows from the end of each preceding range / start of each following range at adjoining boundaries, same semantics as §9.1.

### 9.3 Deferred policies

`random`, `timestamp_ranges`, and any grouped/campaign-aware policy are explicitly deferred past v1. They are not needed to validate the pipeline architecture and real versions of these (for example, a `by_year`-style grouped policy) depend on dataset-specific semantics that belong in private plugin code. When a concrete need exists, add a new registered policy following the same contract in §6.3 — do not retrofit `chronological`/`index_ranges` to grow ad hoc special cases.

Custom group/year/campaign-aware policies remain supported purely through the registration mechanism:

```python
register_split_policy("by_year", split_by_year)
```

---

## 10. Suggested package layout

Implement this as a new subpackage inside the current repository, e.g. `src/preprocessing_core/`. It must not import from `src/anldq/`, and nothing in `src/anldq/` should be expected to import from it during this phase. Extraction into a fully separate public repository is a later, independent step once the architecture has been validated here.

```text
src/preprocessing_core/
  __init__.py
  api.py                 # load_dataset_bundle(path), build_bundle(config), public re-exports
  datasets.py            # BaseDataset, DatasetBundle
  config.py              # YAML parsing and strict schema validation
  pipeline.py            # PreprocessingPipeline orchestration
  registry.py            # three tiny callable registries
  validation.py          # dataset/index/schema validation
  selection.py           # select_time_rows and any future aligned selectors
  loaders/
    __init__.py
    npy.py
    npz.py
    csv.py
    hdf5.py
  transforms/
    __init__.py
    basic.py
    filtering.py
    scaling.py
  splits/
    __init__.py
    chronological.py
    ranges.py
configs/
  example_npz.yaml
  example_csv.yaml
tests/
  preprocessing_core/
    conftest.py
    test_datasets.py
    test_config.py
    test_pipeline.py
    test_transforms.py
    test_splits.py
    test_registration.py
docs/
  preprocessing_core/
    configuration.md
    custom_extensions.md
```

Do not add dataset subclasses, resolver subclasses, builders, CLI commands, or training directories. `BaseDataset` plus the registry is the whole data abstraction for this package.

Every public module in `preprocessing_core` must start with a short module docstring explaining its role in the framework. Public dataclasses, loader/transform/split contracts, and exported functions must have concise docstrings. Comments should explain framework rules and non-obvious edge cases, not restate obvious code.

---

## 11. Registry and extension design

Use three tiny explicit registries. Avoid entry points, auto-discovery, or a plugin manager in v1.

```python
LOADERS: dict[str, Loader] = {}
TRANSFORMS: dict[str, Transform] = {}
SPLIT_POLICIES: dict[str, SplitPolicy] = {}


def register_loader(name: str, fn: Loader) -> None: ...
def register_transform(name: str, fn: Transform) -> None: ...
def register_split_policy(name: str, fn: SplitPolicy) -> None: ...
```

Registration rules:

- reject blank names;
- reject duplicate names by default; allow `replace=True` only for controlled tests;
- include the available registered names in unknown-name errors;
- do not silently overwrite public core operations;
- require experiment code to explicitly import/register its custom operations before calling `load_dataset_bundle()`.

Example plugin use:

```python
# experiment_setup.py
from preprocessing_core import register_loader, register_transform
from my_experiment.loader import load_campaign_data
from my_experiment.transforms import remove_detector_outages

register_loader("campaign", load_campaign_data)
register_transform("remove_detector_outages", remove_detector_outages)
```

Then YAML can use `type: campaign` and `name: remove_detector_outages` without the public package knowing anything about the experiment. This is also where all real, production (SPT/GM2/HLT-style) loaders, transforms, and split policies live — in a private plugin package, not in `preprocessing_core`.

---

## 12. Transform implementation guidance

Start with a small generic transform set. Do not port every current `anldq` transform — most domain-specific ones belong in private plugin code as custom loaders or custom registered transforms.

### Include in first version (pre-split)

- `fill_nan`: replace NaNs with a constant;
- `interpolate_nan_per_channel`: interpolate over timestamps;
- `keep_channels` / `drop_channels`: configured, deterministic channel selection by name/index;
- `keep_features` / `drop_features`: configured, deterministic feature selection by name/index;
- `validate_monotonic_timestamps`: validation-only transform.

### Include in first version (post-split)

- `filter_quantile_timesteps`: time-row filter that uses `select_time_rows`;
- `standard_scale`: fit on train and reuse on selected splits (NumPy only, no sklearn dependency, no estimator objects in state — plain mean/std arrays);
- `robust_scale`: fit on train and reuse on selected splits, same constraints as `standard_scale`.

### Do not include initially

- domain-specific quality criteria that need an externally supplied reference array and train-fit/val-test-reuse state at a pre-split stage (the pattern the current codebase's `select_stable_channels` follows) — this plan does not ship an equivalent, since the pre-split stage has no `fit`/`apply` concept (§7.2); any such selection logic belongs in custom, experiment-specific downstream code, not a public core transform;
- source-specific timestamp trimming;
- rolling filters requiring complicated boundary behavior unless a public test dataset needs them;
- augmentation;
- baseline-removal policy variants;
- transforms that require proprietary packages.

### Scaling example

Scaling is not a separate pipeline stage; it is implemented purely as an ordinary post-split transform using the generic `fit`/`apply` mechanism, exactly like any other post-split transform:

```yaml
post_split:
  - name: standard_scale
    fit: train
    apply: [train, val, test]
    params:
      mode: per_channel_feature
```

Expected behavior:

```text
train: calculate mean/std and store them in state; scale train
val: use saved mean/std; scale val
test: use saved mean/std; scale test
```

Reject val/test execution if state lacks the expected fitted values. Never calculate fallback val/test statistics silently. Fitted mean/std are persisted per §7.4.

### Transform author checklist

Every transform must:

- accept and return `BaseDataset`;
- accept `state` and `fit` keyword arguments even if unused;
- avoid touching `split`, `data_tag`, or `preprocess_steps`;
- call `select_time_rows` for row filtering;
- call `select_channels` for channel filtering;
- update `feature_names` when slicing the feature axis;
- call or rely on pipeline validation after mutation;
- store only serializable fitted state;
- raise clear errors rather than silently falling back.

---

## 13. Error handling and validation

Fail early with useful errors. This package prepares scientific/ML data, so silently wrong alignment is worse than a hard failure.

Use a very small custom exception set:

- `ConfigError`: invalid YAML, invalid schema, unknown registered name, invalid `apply`/`fit`, invalid split config, or any error that should be fixed by editing configuration.
- `PipelineError`: a registered callable failed during execution, returned the wrong type, corrupted orchestrator-owned fields, or produced invalid output.

Lower-level helpers may raise built-in exceptions such as `ValueError`, `KeyError`, `FileNotFoundError`, or `ImportError`. Boundary code should wrap them only when doing so adds useful stage/config context.

Do not build a deep custom exception hierarchy in v1.

Required errors:

- YAML file is missing or invalid;
- unsupported schema version;
- unknown top-level YAML key;
- missing loader, split policy, loader type, transform name, or ratios;
- unknown registered operation, listing available operation names;
- loader returns wrong type or an invalid shape;
- timestamp/label/channel-label length or shape mismatch, or a `metadata["aligned_time"]` array whose first dimension does not match `T`;
- transform returns invalid `BaseDataset`, or writes to an orchestrator-owned field;
- invalid `apply`/`fit` relationship, including any `fit` value other than `"train"`;
- train-fitted state missing for val/test;
- fitted state containing a non-serializable object (e.g. an estimator instance rather than plain arrays);
- invalid, overlapping, out-of-range, or empty split indices — including a split made empty by a `gap` that is too large;
- a transform removes all rows from an applicable split.

`allow_empty` is not a v1 configuration field. In v1, any transform that leaves an applicable split empty raises `PipelineError`. If a future use case needs empty splits, add `allow_empty` with config validation, docs, and tests in a later version.

Every error raised while running a scheduled transform must include full context: the stage (`pre_split`/`post_split`), the transform name, and — for post-split transforms — the split name it was running against, e.g.:

```text
post_split transform 'filter_quantile_timesteps' failed for split 'val': ...
```

Do not catch errors broadly and continue; add this context and re-raise.

### 13.1 Logging policy

Use the standard `logging` module. Library code must never use `print()`.

The package logs stage-level progress and important decisions only:

- `INFO` when a pipeline build starts and finishes;
- `INFO` for selected loader, split policy, and final split sizes;
- `DEBUG` before/after each configured transform, including stage, transform name, split name when relevant, and resulting shape;
- `DEBUG` for fitted-state keys created or reused;
- `WARNING` only for recoverable but suspicious behavior that the user should know about;
- no per-row, per-channel, or inner-loop logs.

Errors should normally be raised, not logged and swallowed. If a caught exception is wrapped with context, use exception chaining (`raise ... from exc`). Avoid duplicate logging at every layer; the top-level caller or CLI can decide whether to log stack traces.

---

## 14. Test plan and acceptance criteria

Tests are part of the feature, not cleanup work after implementation.

### 14.1 Unit tests

1. **`BaseDataset` validation**
   - accepts `(T, C, F)` data;
   - rejects wrong dimensions;
   - rejects timestamp, label, and channel-label length/shape mismatches;
   - accepts `labels=None` and `channel_labels=None` and behaves identically to the labeled case elsewhere in the pipeline;
   - rejects a `metadata["aligned_time"]` array whose first dimension does not match `T`.

2. **Aligned selectors**
   - `select_time_rows` keeps `data`, `timestamps`, `labels`, `channel_labels`, and every `metadata["aligned_time"]` array aligned;
   - `select_channels` keeps `data`, `channel_names`, and the channel axis of `channel_labels` aligned.

3. **YAML schema**
   - valid minimal config parses;
   - unknown key is rejected;
   - invalid `apply`, invalid `fit` (including any value other than `train`), and invalid ratio config are rejected;
   - a `pre_split` entry containing `apply` or `fit` is rejected.

4. **Registries**
   - registration works;
   - duplicate/unknown names fail clearly.

5. **Split policies**
   - chronological boundaries are deterministic;
   - `gap` correctly removes the configured number of rows at each boundary and records the dropped count in metadata;
   - a configuration that would produce an empty split (small `n`, extreme ratios, or an oversized `gap`) raises a clear error;
   - `index_ranges` validates overlap and bounds.

6. **Transforms**
   - row filtering preserves all time alignment;
   - train-fitted scaler uses train statistics for val/test;
   - `apply: [train, val]` leaves test unchanged;
   - missing fitted state fails;
   - fitted state is plain serializable data (no estimator objects);
   - a transform cannot write to `split`/`data_tag`/`preprocess_steps`.

### 14.2 End-to-end tests

Use a tiny synthetic `npz` fixture with known `(T, C, F)` values, timestamps, and channel names. Include at least one variant with `labels` present and one without, to confirm the optional-labels contract holds end to end.

Required end-to-end test:

```text
load once
-> pre_split transform runs once, on the whole dataset
-> chronological split with a gap
-> train/val-only row filtering
-> train-fitted scale reused on val/test
-> DatasetBundle returned
```

Assert all of the following:

- the loader call count is exactly one;
- bundle contains `BaseDataset` objects with `split`/`data_tag` correctly set;
- each output has correct shape/alignment;
- post-split train/val filtering does not change test;
- val/test scaling matches train-fitted parameters, not their own statistics;
- dropped-row counts from the split gap appear in bundle metadata;
- fitted scaler state appears in `bundle.metadata["fitted_state"]`;
- bundle provenance has final split lengths.

### 14.3 Acceptance criteria

The first implementation is done when:

- `load_dataset_bundle(example_yaml)` loads raw data exactly once;
- all output datasets satisfy the `BaseDataset` contract;
- all examples work without proprietary dependencies or paths;
- a custom loader/transform/policy can be registered from external experiment code;
- tests cover alignment, state reuse, selected-split transforms, split validation (including empty-split rejection and gap handling), fitted-state persistence, and one-load behavior;
- there is no training/inference/evaluation code in the package;
- the package does not import from, and is not imported by, `src/anldq/`.

---

## 15. Implementation sequence

Implement in small, testable steps, inside this repository, as a new subpackage. Do not begin by moving large portions of the current `anldq` code.

### Phase 1: Scaffold the new subpackage

1. Create `src/preprocessing_core/` with the layout in §10.
2. Add a dedicated `tests/preprocessing_core/` directory and pytest setup.
3. Use minimal required dependencies:

```toml
dependencies = [
  "numpy",
  "pandas",
  "PyYAML",
]

[project.optional-dependencies]
hdf5 = ["h5py"]
```

Keep `pytest` in a development dependency group. `h5py` is an optional extra, not a hard dependency, since only the `hdf5` loader needs it and most example configs use `npy`/`npz`/`csv`. Do not add PyTorch, scikit-learn, or any `anldq` dependency to this subpackage unless a concrete public feature requires it.

### Phase 2: `BaseDataset` and selectors

1. Implement `BaseDataset` and `DatasetBundle`.
2. Implement `validate()` and `copy()`.
3. Implement `select_time_rows`.
4. Write tests proving alignment correctness, and proving that transforms cannot write to orchestrator-owned fields, before adding YAML.

### Phase 3: Registries and built-in operations

1. Implement the three registries.
2. Implement `npz` first as the test/default loader.
3. Implement `chronological` first as the test/default policy, including `gap` handling and empty-split rejection.
4. Implement `fill_nan` and a basic split-specific row filter.
5. Add registry and operation tests.

### Phase 4: Strict YAML configuration

1. Define dataclasses for `LoaderConfig`, `TransformConfig`, `SplitConfig`, and `PipelineConfig`.
2. Parse YAML with `yaml.safe_load`.
3. Validate the exact schema and reject unknown keys, including rejecting `apply`/`fit` on `pre_split` entries and any `fit` value other than `"train"`.
4. Add `configs/example_npz.yaml` with no absolute or private paths.
5. Test malformed configurations.

### Phase 5: Single-pass pipeline and bundle

1. Implement `PreprocessingPipeline` exactly once in `pipeline.py`.
2. Ensure the loader is called once.
3. Run pre-split transforms once, on the whole dataset, with no `apply`/`fit`.
4. Generate and validate split indices, including gap and empty-split handling.
5. Materialize three `BaseDataset` instances with `split`/`data_tag` set by the pipeline.
6. Run each post-split transform in deterministic train/val/test order, with full error context (stage, transform, split).
7. Return the bundle via `build_bundle(config)`.
8. Add the end-to-end test before proceeding.

### Phase 6: Train-fitted state and persistence

1. Add `fit: train` validation (rejecting any other value) and state scoping.
2. Implement `standard_scale` and `robust_scale` using NumPy only.
3. Add robust error messages for missing or malformed state, and for non-serializable fitted state.
4. Record fitted state in `bundle.metadata["fitted_state"]`.
5. Decide and implement the on-disk persistence format for fitted state (e.g. alongside other pipeline output artifacts) so it can be reapplied to new data later.
6. Test that val/test do not fit their own parameters, and that persisted state round-trips correctly.

### Phase 7: More generic public features

1. Add CSV loader using pandas.
2. Add the minimal `hdf5` loader using `h5py` (optional dependency, §6.1). This was pulled forward into v1 scope earlier in this plan (§1) rather than deferred; it is sequenced here, alongside the CSV loader, simply because both are "one more generic format reader" work and neither blocks the Phase 1-6 architecture.
3. Add `index_ranges` policy with gap handling.
4. Add `keep_channels`/`drop_channels` pre-split transforms (via `select_channels`, §6.2) and `keep_features`/`drop_features` pre-split transforms (direct feature-axis slicing of `data`/`feature_names`; no dedicated helper is needed since no other field is feature-aligned).
5. Write clear docs for custom registrations.

### Phase 8: Documentation and review

1. README section: the one-load pipeline diagram, installation, and minimal example.
2. `docs/preprocessing_core/configuration.md`: full YAML reference and transform scheduling rules.
3. `docs/preprocessing_core/custom_extensions.md`: loader/transform/policy contracts and registration examples, with explicit guidance that production, experiment-specific logic (custom loaders doing domain-specific channel/quality selection, custom split policies) belongs in a private plugin package, not in `preprocessing_core`.
4. Confirm the package has zero imports from `src/anldq/` and zero proprietary paths/dependencies.
5. Run tests in a clean environment.

Extraction of this subpackage into a fully independent public repository (per the separate, preliminary `PUBLIC_CORE_MIGRATION_PLAN.md`) is an explicit, later, follow-on step and is out of scope for this implementation sequence.

---

## 16. Migration reference: current code to new core

This table is guidance only. Copy behavior selectively, not files wholesale, and do not import from `src/anldq/`.

| Current location | Useful concept | New location / action |
| --- | --- | --- |
| `datasets/base.py` | `(T, C, D)` convention and aligned arrays | `preprocessing_core/datasets.py`, `selection.py`, `validation.py` |
| `BaseDataset._build()` | ordered pipeline idea | replace with `PreprocessingPipeline.build()` |
| `BaseDataset._run_transform_steps()` | named transform execution plus `apply` / `fit` state | simplify in `pipeline.py`; `fit` restricted to `"train"` only |
| `datasets/builder.py` | config-to-build orchestration | replace; do not build three dataset instances independently |
| `process/split_time.py` | split strategy behavior; confirms all policies are NumPy-only | small pure functions under `preprocessing_core/splits/`; only `chronological`/`index_ranges` in v1 |
| `process/filters.py` | generic filtering concepts (`filter_quantile_timesteps`) | port only the generic, non-domain-specific ones |
| `process/transforms.py` | NaN interpolation / channel-feature helpers | port selectively |
| `datasets/registry.py` | name-to-callable lookup | `preprocessing_core/registry.py`, with three focused registries |
| YAML SPT configs | `apply` and `fit` configuration examples | sanitize/generalize; do not copy data names, paths, or SPT-specific transform semantics |
| family-specific subclasses/loaders (GM2, SPT, HLT) | proprietary/source-specific behavior, including complex multi-rule quality filtering | private plugin package, registered via `register_loader`/`register_transform`/`register_split_policy`; not part of this plan's scope |

Note: SPT in particular embeds roughly a thousand lines of domain-specific loading, quality-control, and custom-scaling logic directly in one dataset subclass today. None of that is addressed by this plan — it is explicitly deferred to private plugin code written against the registration API once this core exists. Decomposing it is a separate, later effort, not a prerequisite for implementing this plan.

---

## 17. Risks and explicit non-goals

### Risks

1. **Memory:** one full `(T, C, F)` array plus three split copies can require substantial memory for large time-series datasets. The v1 copying approach is safer for isolation/alignment. If real workloads (in a private plugin, with real data) prove memory-limited, add an explicitly tested view/index representation later; do not design around this prematurely.
2. **Leakage:** moving a transform from post-split to pre-split changes its scientific meaning. Documentation must clearly state that train-fitted or train-only transforms belong in `post_split`, and that pre-split transforms are, by construction, whole-dataset operations.
3. **Transform quality:** external transforms can violate alignment if they manipulate raw arrays directly instead of going through `select_time_rows`. The `BaseDataset` contract and that helper are the documented path.
4. **Scope creep via custom loaders:** pushing all experiment-specific selection logic into custom loaders is a deliberate simplification, but it means the public core cannot validate that a custom loader's internal channel/metadata handling is self-consistent — that responsibility sits entirely with the loader author. Document this tradeoff clearly in `custom_extensions.md`.
5. **Fitted-state persistence format is not yet decided** (§7.4, Phase 6). This is flagged explicitly as an open implementation decision, not a design gap to be silently resolved.

### Explicit non-goals for v1

- No `BaseDataset` inheritance hierarchy.
- No second data-container type (`DataBlock`) alongside `BaseDataset`.
- No train/infer/eval APIs.
- No automatic loader/transform discovery.
- No backward compatibility with `anldq.datasets`.
- No private dataset migration.
- No Parquet loader in v1 (HDF5 is in v1, §6.1).
- No split policies beyond `chronological`, `index_ranges`, and `seasonal_block` in v1.
- No channel/feature selection transform complexity beyond simple, deterministic keep/drop lists.
- No optimization beyond eliminating repeated full raw loads and repeated pre-split transforms.
- No large inline implementation blocks inside `PreprocessingPipeline`; complex validation, metadata construction, selection, state serialization, and operation-specific behavior must live in focused helper functions or modules.

---

## 18. Final intended user experience

A public user should be able to understand the pipeline from one config:

```text
one loader
-> pre_split transforms on all raw data (no apply/fit)
-> one split policy, with optional leakage-reducing gaps
-> post_split transforms on selected splits (apply/fit supported; scaling is just one such transform)
-> DatasetBundle(train, val, test, metadata)
```

And call it with one clear API:

```python
from preprocessing_core import load_dataset_bundle

bundle = load_dataset_bundle("configs/example_npz.yaml")
print(bundle.train.data.shape)
print(bundle.train.n_channels, bundle.train.n_features)
print(bundle.val.timestamps)
print(bundle.test.labels)             # None if the source experiment has no per-timestep labels
print(bundle.test.channel_labels)     # None if the source experiment has no per-channel labels
print(bundle.metadata["fitted_state"])
```

This is the foundation for later training and inference work. `bundle.train`/`bundle.val`/`bundle.test` are ordinary `BaseDataset` instances, and `BaseDataset`/`DatasetBundle` describe **data only** — they are not designed to be a drop-in replacement for the current `anldq.datasets.base.BaseDataset` at every existing call site, and they do not carry evaluation concerns such as score aggregation or channel grouping as first-class dataset fields (§5).

Downstream training and inference are expected to consume this bundle through their own result types, not by reading `anldq`-specific attributes off `BaseDataset` itself. For example (illustrative, not part of this package):

```python
@dataclass
class TrainResult:
    run_id: str
    checkpoint_path: Path
    loss_history: dict
    val_loss_history: dict
    best_epoch: int | None
    best_val_loss: float | None
    output_dir: Path

@dataclass
class InferenceResult:
    errors: np.ndarray
    predictions: np.ndarray
    scores: np.ndarray
    timestamps: np.ndarray | None
    labels: np.ndarray | None
    feature_names: list[str] | None
    metadata: dict
    output_dir: Path | None
```

A future `Trainer`/`Infer`-equivalent reads `bundle.train`/`bundle.val`/`bundle.test` for data (`.data`, `.timestamps`, `.labels`, `.channel_names`, `.feature_names`) and produces its own `TrainResult`/`InferenceResult` objects; any score aggregation, channel-group-aware reduction, or other evaluation logic lives in that downstream code and its own config, not on `BaseDataset`. This is a deliberate, narrower claim than "works with the existing `anldq` trainer/infer code with no adapter layer": wiring this bundle into the *existing* `anldq` trainer/infer CLI entry points (`src/anldq/trainer`, `src/anldq/infer`, `src/anldq/cli`), and building out real (private) loaders, transforms, and split policies for SPT/GM2/HLT, remain explicit, separate follow-on migration efforts and are not part of this plan.
