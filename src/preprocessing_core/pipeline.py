"""Pipeline orchestration for configured preprocessing builds."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from preprocessing_core.config import PipelineConfig, load_config
from preprocessing_core.datasets import BaseDataset, DatasetBundle
from preprocessing_core.registry import Loader, SplitPolicy, get_loader, get_split_policy
from preprocessing_core.selection import select_time_rows
from preprocessing_core.validation import PipelineError, validate_split_indices

LOGGER = logging.getLogger(__name__)
SPLIT_ORDER = ("train", "val", "test")


@dataclass
class PreprocessingPipeline:
    """Thin orchestrator for one configured preprocessing pipeline."""

    config: PipelineConfig
    loader: Loader | None = None
    split_policy: SplitPolicy | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "PreprocessingPipeline":
        """Create a pipeline from a YAML config file."""
        return cls.from_config(load_config(path))

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "PreprocessingPipeline":
        """Create a pipeline from validated config and resolve registered callables."""
        config.validate()
        pipeline = cls(config=config)
        pipeline.resolve()
        return pipeline

    def resolve(self) -> None:
        """Resolve configured loader and split policy names."""
        self.loader = get_loader(self.config.loader.type)
        self.split_policy = get_split_policy(self.config.split.policy)

    def build(self) -> DatasetBundle:
        """Run the configured one-load preprocessing pipeline."""
        if self.config.pre_split or self.config.post_split:
            raise NotImplementedError("Pipeline transforms are not implemented yet")

        LOGGER.info(
            "Starting preprocessing build with loader %r and split policy %r",
            self.config.loader.type,
            self.config.split.policy,
        )

        full = self._load_dataset()
        splits = self._split_dataset(full)
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
        if self.loader is None:
            raise PipelineError("Pipeline loader has not been resolved")

        try:
            dataset = self.loader(**self.config.loader.params)
        except Exception as exc:
            raise PipelineError(f"Loader {self.config.loader.type!r} failed") from exc

        if not isinstance(dataset, BaseDataset):
            raise PipelineError(f"Loader {self.config.loader.type!r} did not return BaseDataset")

        dataset.validate()
        LOGGER.debug("Loaded dataset with shape %s", dataset.shape)
        return dataset

    def _split_dataset(self, full: BaseDataset) -> dict[str, BaseDataset]:
        """Run the configured split policy and materialize split datasets."""
        if self.split_policy is None:
            raise PipelineError("Pipeline split policy has not been resolved")

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

    def _make_output(self, splits: dict[str, BaseDataset]) -> DatasetBundle:
        """Return the final dataset bundle."""
        return DatasetBundle(
            train=splits["train"],
            val=splits["val"],
            test=splits["test"],
            metadata=dict(self.config.metadata),
        )
