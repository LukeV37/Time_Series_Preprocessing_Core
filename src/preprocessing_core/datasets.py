"""Data containers for the preprocessing core."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BaseDataset:
    """Canonical container for one full or split time-series dataset.

    ``BaseDataset`` stores data in ``(T, C, F)`` form, where ``T`` is the
    number of time rows, ``C`` is the number of channels, and ``F`` is the
    number of features. Optional arrays and names are validated against these
    dimensions so loaders, transforms, and split policies can share a common
    dataset contract.

    Attributes
    ----------
    data : np.ndarray
        Time-series data with shape ``(T, C, F)``. ``T`` must be non-zero.
    timestamps : np.ndarray | None
        Optional time-aligned timestamps with shape ``(T,)``. The default is
        ``None``.
    labels : np.ndarray | None
        Optional time-aligned labels with shape ``(T,)``. The default is
        ``None``.
    channel_labels : np.ndarray | None
        Optional time- and channel-aligned labels with shape ``(T, C)``. The
        default is ``None``.
    channel_names : list[str] | None
        Optional names for the ``C`` channels. The default is ``None``.
    feature_names : list[str] | None
        Optional names for the ``F`` features. The default is ``None``.
    metadata : dict[str, Any]
        Free-form dataset metadata. If ``metadata["aligned_time"]`` is set,
        its values must be time-aligned with the first dimension ``T``.
    split : str | None
        Pipeline-owned split name, usually one of ``{"train", "val", "test"}``.
        The default is ``None``.
    data_tag : str | None
        Optional pipeline-owned tag identifying the dataset source. The default
        is ``None``.
    preprocess_steps : list[str]
        Provenance labels for preprocessing steps applied to this dataset.

    Raises
    ------
    PipelineError
        Raised by :meth:`validate` when data shape or aligned fields violate
        the dataset contract.
    """

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
        """Return a non-aliased copy safe for split-specific mutation.

        Returns
        -------
        BaseDataset
            Deep copy of this dataset with arrays, lists, metadata, and
            provenance copied so split-specific mutation does not alias the
            source dataset.
        """
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
        """Return a copy tagged with orchestrator-owned split fields.

        Returns
        -------
        BaseDataset
            Copied dataset whose ``split`` and ``data_tag`` fields are set to
            the provided values.
        """
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
    """Container for the canonical train, validation, and test datasets.

    ``DatasetBundle`` is the final output of a preprocessing build. It groups
    the three split datasets with metadata that applies to the bundle as a
    whole, such as fitted transform state recorded by the pipeline.

    Attributes
    ----------
    train : BaseDataset
        Training split dataset.
    val : BaseDataset
        Validation split dataset.
    test : BaseDataset
        Test split dataset.
    metadata : dict[str, Any]
        Shared bundle-level metadata. Pipelines store fitted transform state
        under ``metadata["fitted_state"]`` when applicable.
    """

    train: BaseDataset
    val: BaseDataset
    test: BaseDataset
    metadata: dict[str, Any] = field(default_factory=dict)

    def splits(self) -> dict[str, BaseDataset]:
        """Return split datasets keyed by split name.

        Returns
        -------
        dict[str, BaseDataset]
            Mapping with ``"train"``, ``"val"``, and ``"test"`` keys.
        """
        return {"train": self.train, "val": self.val, "test": self.test}
