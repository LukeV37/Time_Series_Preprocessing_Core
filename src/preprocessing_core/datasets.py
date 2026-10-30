"""Data containers for the preprocessing core."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BaseDataset:
    """One full or split time-series dataset in canonical ``(T, C, F)`` form."""

    data: np.ndarray
    timestamps: np.ndarray | None = None
    labels: np.ndarray | None = None
    channel_labels: np.ndarray | None = None
    channel_names: list[str] | None = None
    feature_names: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    split: str | None = None
    data_tag: str | None = None
    preprocess_steps: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Raise if data or aligned fields violate the dataset contract."""
        from preprocessing_core.validation import validate_dataset

        validate_dataset(self)

    def copy(self) -> BaseDataset:
        """Return a non-aliased copy safe for split-specific mutation."""
        return BaseDataset(
            data=np.array(self.data, copy=True),
            timestamps=None if self.timestamps is None else np.array(self.timestamps, copy=True),
            labels=None if self.labels is None else np.array(self.labels, copy=True),
            channel_labels=None
            if self.channel_labels is None
            else np.array(self.channel_labels, copy=True),
            channel_names=None if self.channel_names is None else list(self.channel_names),
            feature_names=None if self.feature_names is None else list(self.feature_names),
            metadata=copy.deepcopy(self.metadata),
            split=self.split,
            data_tag=self.data_tag,
            preprocess_steps=list(self.preprocess_steps),
        )

    def with_split(self, split: str, data_tag: str | None = None) -> BaseDataset:
        """Return a copy tagged with orchestrator-owned split fields."""
        result = self.copy()
        result.split = split
        result.data_tag = data_tag
        return result

    def __len__(self) -> int:
        return int(self.data.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.data.shape[2])


@dataclass
class DatasetBundle:
    """Train/validation/test datasets plus shared preprocessing metadata."""

    train: BaseDataset
    val: BaseDataset
    test: BaseDataset
    metadata: dict[str, Any] = field(default_factory=dict)

    def splits(self) -> dict[str, BaseDataset]:
        """Return split datasets keyed by split name."""
        return {"train": self.train, "val": self.val, "test": self.test}
