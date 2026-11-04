"""Pipeline orchestration for configured preprocessing builds."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

from preprocessing_core.config import PipelineConfig, TransformConfig
from preprocessing_core.datasets import BaseDataset, DatasetBundle
from preprocessing_core.registry import (
    Loader,
    SplitPolicy,
    Transform,
    get_loader,
    get_split_policy,
    get_transform,
)
from preprocessing_core.selection import select_time_rows
from preprocessing_core.validation import (
    PipelineError,
    validate_split_indices,
    validate_state_is_serializable,
)

LOGGER = logging.getLogger(__name__)
SPLIT_ORDER = ("train", "val", "test")


@dataclass
class TransformStep:
    """Resolved transform step ready for pipeline execution."""

    stage: str
    position: int
    name: str
    fn: Transform
    params: dict[str, Any]
    apply: tuple[str, ...]
    fit: str | None
    state_key: str

    @property
    def label(self) -> str:
        """Return the provenance label recorded on transformed datasets."""
        return f"{self.stage}:{self.position}:{self.name}"


@dataclass
class PreprocessingPipeline:
    """Orchestrator for one configured preprocessing build.

    A pipeline loads the full dataset once, applies full-dataset transforms,
    computes the train/validation/test split indices, materializes split
    datasets, and then applies split-specific transforms in deterministic
    ``train``, ``val``, ``test`` order. Fitted post-split transform state is
    retained during the build and copied into the returned DatasetBundle.

    Attributes
    ----------
    config : PipelineConfig
        Validated pipeline configuration describing the loader, split policy,
        transforms, and user metadata.
    loader : Loader
        Resolved loader callable.
    split_policy : SplitPolicy
        Resolved split policy callable.
    pre_split_steps : list[TransformStep]
        Resolved transform steps run before splitting.
    post_split_steps : list[TransformStep]
        Resolved transform steps run after splitting.
    fitted_state : dict[str, dict[str, object]]
        Mutable state captured for fitted post-split transforms during build().

    Raises
    ------
    PipelineError
        Raised during build() if loading or transforms fail, a callable returns
        an invalid object, or fitted transform state cannot be serialized.
    """

    config: PipelineConfig
    loader: Loader
    split_policy: SplitPolicy
    pre_split_steps: list[TransformStep]
    post_split_steps: list[TransformStep]
    fitted_state: dict[str, dict[str, object]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: PipelineConfig) -> PreprocessingPipeline:
        """Create a fully resolved pipeline that is ready to build.

        Returns
        -------
        PreprocessingPipeline
            Pipeline initialized with ``config`` and resolved loader, split
            policy, and transform callables.
        """
        config.validate()
        return cls(
            config=config,
            loader=get_loader(config.loader.type),
            split_policy=get_split_policy(config.split.policy),
            pre_split_steps=_resolve_transform_steps(config.pre_split, stage="pre_split"),
            post_split_steps=_resolve_transform_steps(config.post_split, stage="post_split"),
        )

    def build(self) -> DatasetBundle:
        """Run the configured one-load preprocessing pipeline.

        Returns
        -------
        DatasetBundle
            Train, validation, and test datasets plus bundle-level metadata,
            including any fitted post-split transform state.
        """
        self.fitted_state = {}

        LOGGER.info(
            "Starting preprocessing build with loader %r and split policy %r",
            self.config.loader.type,
            self.config.split.policy,
        )

        full = self._load_dataset()
        full = self._run_pre_split_transforms(full)
        splits = self._split_dataset(full)
        splits = self._run_post_split_transforms(splits)
        bundle = self._make_output(splits)

        LOGGER.info(
            "Finished preprocessing build: train=%s val=%s test=%s",
            bundle.train.shape,
            bundle.val.shape,
            bundle.test.shape,
        )
        return bundle

    def _load_dataset(self) -> BaseDataset:
        """Load and validate the full dataset exactly once."""
        try:
            dataset = self.loader(**self.config.loader.params)
        except Exception as exc:
            raise PipelineError(f"Loader {self.config.loader.type!r} failed") from exc

        if not isinstance(dataset, BaseDataset):
            raise PipelineError(f"Loader {self.config.loader.type!r} did not return BaseDataset")

        dataset.validate()
        LOGGER.debug("Loaded dataset with shape %s", dataset.shape)
        return dataset

    def _run_pre_split_transforms(self, dataset: BaseDataset) -> BaseDataset:
        """Run full-dataset transforms before split indices are computed."""
        result = dataset
        for step in self.pre_split_steps or []:
            result = self._run_transform_step(result, step, state={}, fit=False)
        return result

    def _split_dataset(self, full: BaseDataset) -> dict[str, BaseDataset]:
        """Run the configured split policy and materialize split datasets."""
        try:
            indices = self.split_policy(full, **self.config.split.params)
        except Exception as exc:
            raise PipelineError(f"Split policy {self.config.split.policy!r} failed") from exc

        validate_split_indices(indices, n_rows=len(full), chronological_expected=False)
        data_tag = self.config.metadata.get("dataset_name")

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

    def _run_post_split_transforms(self, splits: dict[str, BaseDataset]) -> dict[str, BaseDataset]:
        """Run split-specific transforms in deterministic train/val/test order."""
        for step in self.post_split_steps or []:
            state = self._state_for_step(step)
            for split_name in SPLIT_ORDER:
                if split_name not in step.apply:
                    continue
                splits[split_name] = self._run_transform_step(
                    splits[split_name],
                    step,
                    state=state,
                    fit=step.fit == split_name,
                )
            if step.fit is not None:
                validate_state_is_serializable(state, context=f"state for {step.label}")
        return splits

    def _run_transform_step(
        self,
        dataset: BaseDataset,
        step: TransformStep,
        *,
        state: dict[str, object],
        fit: bool,
    ) -> BaseDataset:
        """Run one transform with validation and error context."""
        original_split = dataset.split
        original_data_tag = dataset.data_tag
        try:
            result = step.fn(dataset, state=state, fit=fit, **step.params)
        except Exception as exc:
            raise PipelineError(f"Transform {step.name!r} failed") from exc

        if not isinstance(result, BaseDataset):
            raise PipelineError(f"Transform {step.name!r} did not return BaseDataset")

        result.split = original_split
        result.data_tag = original_data_tag
        result.preprocess_steps = list(dataset.preprocess_steps) + [step.label]
        result.validate()
        return result

    def _state_for_step(self, step: TransformStep) -> dict[str, object]:
        """Return the mutable state dict for one post-split transform step."""
        if step.fit is None:
            return {}
        return self.fitted_state.setdefault(step.state_key, {})

    def _make_output(self, splits: dict[str, BaseDataset]) -> DatasetBundle:
        """Return the final dataset bundle."""
        metadata = dict(self.config.metadata)
        metadata["fitted_state"] = copy.deepcopy(self.fitted_state or {})
        return DatasetBundle(
            train=splits["train"],
            val=splits["val"],
            test=splits["test"],
            metadata=metadata,
        )


def _resolve_transform_steps(configs: list[TransformConfig], *, stage: str) -> list[TransformStep]:
    """Resolve transform configs into executable steps."""
    steps = []
    for position, config in enumerate(configs):
        apply = config.apply or SPLIT_ORDER
        steps.append(
            TransformStep(
                stage=stage,
                position=position,
                name=config.name,
                fn=get_transform(config.name),
                params=dict(config.params),
                apply=apply,
                fit=config.fit,
                state_key=config.state_key or f"{stage}:{position}:{config.name}",
            )
        )
    return steps
