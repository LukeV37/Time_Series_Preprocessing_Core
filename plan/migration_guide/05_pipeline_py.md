# Migration Map: `src/preprocessing_core/pipeline.py`

## Purpose

Implement the mini-framework's orchestration algorithm.

`PreprocessingPipeline` is allowed to be a framework object. It should abstract away scheduling and execution rules for downstream users. However, it must stay thin: it sequences operations and delegates decisions to helpers.

## Scope

Owns:

- one-load pipeline sequence
- resolved callable schedule
- build-time fitted state dictionary
- stage-level logging
- wrapping transform errors with context
- producing `DatasetBundle`

Delegates:

- YAML parsing to `config.py`
- callable lookup to `registry.py`
- dataset validation to `BaseDataset.validate()` / `validation.py`
- split index validation to `validation.py`
- row/channel selection to `selection.py`
- operation behavior to loaders/transforms/split policies
- metadata assembly to `make_bundle_metadata()` helper
- fitted-state serializability checks to validation helper

## Public shape

```python
"""Pipeline orchestration for configured preprocessing builds."""

from dataclasses import dataclass
from typing import Callable
import logging

from preprocessing_core.config import PipelineConfig, TransformConfig, load_config
from preprocessing_core.datasets import BaseDataset, DatasetBundle
from preprocessing_core.registry import get_loader, get_split_policy, get_transform
from preprocessing_core.selection import select_time_rows
from preprocessing_core.validation import (
    ConfigError,
    PipelineError,
    validate_apply_and_fit,
    validate_split_indices,
    validate_state_is_serializable,
)

LOGGER = logging.getLogger(__name__)
SPLIT_ORDER = ("train", "val", "test")
```

## `ScheduledStep`

```python
@dataclass(frozen=True)
class ScheduledStep:
    """Resolved transform step ready for pipeline execution."""

    stage: str
    position: int
    name: str
    fn: Callable[..., BaseDataset]
    params: dict[str, object]
    apply: tuple[str, ...]
    fit: str | None
    state_key: str
```

Keep this as a data object. Do not add execution methods.

## `PreprocessingPipeline` pseudocode

```python
class PreprocessingPipeline:
    """Thin orchestrator for one configured preprocessing pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.loader = None
        self.split_policy = None
        self.pre_split_steps: list[ScheduledStep] = []
        self.post_split_steps: list[ScheduledStep] = []
        self.state: dict[str, dict[str, object]] = {}
        self._split_metadata: dict[str, object] = {}

    @classmethod
    def from_yaml(cls, path: str) -> "PreprocessingPipeline":
        return cls.from_config(load_config(path))

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "PreprocessingPipeline":
        pipeline = cls(config)
        pipeline.resolve()
        return pipeline

    def resolve(self) -> None:
        """Resolve configured names to callables and executable schedules."""
        self.loader = get_loader(self.config.loader.type)
        self.split_policy = get_split_policy(self.config.split.policy)
        self.pre_split_steps = schedule_transform_steps(
            stage="pre_split",
            step_configs=self.config.pre_split,
            allow_apply=False,
            allow_fit=False,
        )
        self.post_split_steps = schedule_transform_steps(
            stage="post_split",
            step_configs=self.config.post_split,
            allow_apply=True,
            allow_fit=True,
        )

    def build(self) -> DatasetBundle:
        """Run the complete one-load preprocessing pipeline."""
        LOGGER.info("Starting preprocessing build")
        LOGGER.info("Using loader %r and split policy %r", self.config.loader.type, self.config.split.policy)

        full = self._load_dataset()
        full = self._run_pre_split(full)
        splits = self._split_dataset(full)
        splits = self._run_post_split(splits)
        bundle = self._make_output(splits)

        LOGGER.info(
            "Finished preprocessing build: train=%s val=%s test=%s",
            bundle.train.shape,
            bundle.val.shape,
            bundle.test.shape,
        )
        return bundle
```

## Private methods

Keep each method short. If a method starts doing detailed validation or metadata manipulation, extract a helper.

```python
    def _load_dataset(self) -> BaseDataset:
        try:
            dataset = self.loader(**self.config.loader.params)
        except Exception as exc:
            raise PipelineError(f"Loader {self.config.loader.type!r} failed") from exc

        if not isinstance(dataset, BaseDataset):
            raise PipelineError(f"Loader {self.config.loader.type!r} did not return BaseDataset")

        dataset.validate()
        LOGGER.debug("Loaded dataset with shape %s", dataset.shape)
        return dataset

    def _run_pre_split(self, dataset: BaseDataset) -> BaseDataset:
        for step in self.pre_split_steps:
            LOGGER.debug("Running pre_split transform %r", step.name)
            dataset = run_transform_step(step=step, dataset=dataset, state={}, fit=False)
            dataset.validate()
            LOGGER.debug("Finished pre_split transform %r: shape=%s", step.name, dataset.shape)
        return dataset

    def _split_dataset(self, full: BaseDataset) -> dict[str, BaseDataset]:
        try:
            indices = self.split_policy(full, **self.config.split.params)
        except Exception as exc:
            raise PipelineError(f"Split policy {self.config.split.policy!r} failed") from exc

        validate_split_indices(indices, n_rows=len(full), chronological_expected=False)
        data_tag = self.config.metadata.get("dataset_name")
        self._split_metadata = compute_builtin_split_metadata(
            policy_name=self.config.split.policy,
            dataset=full,
            params=self.config.split.params,
            indices=indices,
        )

        splits = {
            name: select_time_rows(full, indices[name]).with_split(name, data_tag)
            for name in SPLIT_ORDER
        }
        for dataset in splits.values():
            dataset.validate()

        LOGGER.info(
            "Created splits: train=%d val=%d test=%d",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )
        return splits

    def _run_post_split(self, splits: dict[str, BaseDataset]) -> dict[str, BaseDataset]:
        for step in self.post_split_steps:
            step_state = self.state.setdefault(step.state_key, {})
            for split_name in SPLIT_ORDER:
                if split_name not in step.apply:
                    continue
                LOGGER.debug("Running post_split transform %r for split %r", step.name, split_name)
                splits[split_name] = run_transform_step(
                    step=step,
                    dataset=splits[split_name],
                    state=step_state,
                    fit=step.fit == split_name,
                    split_name=split_name,
                )
                splits[split_name].validate()
                if len(splits[split_name]) == 0:
                    raise PipelineError(
                        f"post_split transform {step.name!r} left split {split_name!r} empty"
                    )
                LOGGER.debug(
                    "Finished post_split transform %r for split %r: shape=%s",
                    step.name,
                    split_name,
                    splits[split_name].shape,
                )
            validate_state_is_serializable(step_state, context=f"state for {step.state_key!r}")
        return splits

    def _make_output(self, splits: dict[str, BaseDataset]) -> DatasetBundle:
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
                split_metadata=self._split_metadata,
            ),
        )
```

Use a small helper such as `compute_builtin_split_metadata(...)` rather than a richer split-result abstraction in v1. Built-in policies still return only plain split-index dicts. The helper may return `{}` for policies that have no extra provenance, and for built-ins with leakage gaps it can add simple counts such as `{"dropped_rows": {...}}`.

## Recommendation: `split_metadata` handoff

Best practice here is to keep split metadata as **pipeline-owned ephemeral state** and store it on `self._split_metadata` between `_split_dataset()` and `_make_output()`.

Why this is the best v1 choice:

- it keeps `_split_dataset()`'s public shape simple: it still returns only `dict[str, BaseDataset]`;
- it preserves the plan's intended split-policy contract of **indices only**, with provenance computed by a pipeline-owned helper rather than by a richer split result object;
- it avoids introducing a small ad hoc tuple return like `(splits, split_metadata)` that every caller has to remember;
- it keeps provenance ownership in one place: pipeline orchestration computes bundle metadata, while split policies compute only indices.

Avoid introducing a `SplitResult` abstraction in v1 just to carry a small amount of metadata. That is extra framework surface without a clear need today.

Guardrails:

- initialize `self._split_metadata = {}` in `__init__`;
- overwrite it exactly once per `build()` in `_split_dataset()`;
- treat it as private pipeline state, not as a general mutable scratchpad for transforms.

## Helper functions in this file

These helpers may live in `pipeline.py` if they stay small. If they grow, move them to dedicated modules.

### `schedule_transform_steps`

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
        if not allow_apply and step_config.apply is not None:
            raise ConfigError(f"{stage} step {step_config.name!r} cannot use apply")
        if not allow_fit and step_config.fit is not None:
            raise ConfigError(f"{stage} step {step_config.name!r} cannot use fit")

        apply = tuple(step_config.apply or SPLIT_ORDER)
        fit = step_config.fit
        validate_apply_and_fit(stage=stage, name=step_config.name, apply=apply, fit=fit)

        scheduled.append(
            ScheduledStep(
                stage=stage,
                position=position,
                name=step_config.name,
                fn=get_transform(step_config.name),
                params=dict(step_config.params or {}),
                apply=apply,
                fit=fit,
                state_key=step_config.state_key or f"{stage}:{position}:{step_config.name}",
            )
        )
    return scheduled
```

### `run_transform_step`

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

    before = snapshot_orchestrator_fields(dataset)
    try:
        result = step.fn(dataset, state=state, fit=fit, **step.params)
    except Exception as exc:
        raise PipelineError(f"{context} failed") from exc

    if not isinstance(result, BaseDataset):
        raise PipelineError(f"{context} did not return BaseDataset")

    after = snapshot_orchestrator_fields(result)
    if after != before:
        raise PipelineError(f"{context} modified orchestrator-owned fields")

    return result
```

### `make_bundle_metadata`

```python
def make_bundle_metadata(...):
    return {
        **dict(config.metadata),
        "config_path": config.config_path,
        "config_version": config.version,
        "canonical_shape": "(T, C, F)",
        "loader": config.loader.type,
        "split_policy": config.split.policy,
        "split_parameters": dict(config.split.params),
        "pre_split_steps": [step.name for step in pre_split_steps],
        "post_split_steps": {
            name: [step.name for step in post_split_steps if name in step.apply]
            for name in SPLIT_ORDER
        },
        "final_shapes": {name: tuple(dataset.shape) for name, dataset in splits.items()},
        "final_lengths": {name: len(dataset) for name, dataset in splits.items()},
        "fitted_state": copy_serializable_state(fitted_state),
    }
```

## Logging policy

Use standard `logging`.

- `INFO`: build start/end, selected loader/split policy, final split sizes.
- `DEBUG`: each transform start/end, shape after transform, fitted state key creation/reuse.
- `WARNING`: only recoverable suspicious behavior. Avoid warnings for expected choices.
- Never `print()`.
- Do not log and swallow exceptions.

## Error handling

- Use `PipelineError` for runtime failures and wrong callable returns.
- Use `ConfigError` for invalid scheduling/config combinations.
- Always chain caught exceptions with `raise ... from exc`.
- Transform errors must include stage, transform name, and split when applicable.

## Human readability

- `build()` should read like the pipeline diagram.
- Each private method should fit on one screen if possible.
- Complex details belong in helpers.
- Do not add old `anldq` builder compatibility here.

## Tests

- loader is called exactly once
- pre-split transforms run once
- post-split transforms run in train/val/test order
- `fit: train` state is available to val/test
- transform errors include context
- misbehaving transform cannot change `split`/`data_tag`/`preprocess_steps`
- bundle metadata contains provenance and fitted state
