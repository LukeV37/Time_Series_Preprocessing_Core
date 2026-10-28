import numpy as np
import pytest

from preprocessing_core import BaseDataset, DatasetBundle, PipelineError


def make_dataset() -> BaseDataset:
    return BaseDataset(
        data=np.ones((4, 2, 1)),
        timestamps=np.arange(4),
        labels=np.array([0, 1, 0, 1]),
        channel_labels=np.zeros((4, 2)),
        channel_names=["a", "b"],
        feature_names=["value"],
        metadata={"aligned_time": {"file_id": np.array([1, 1, 2, 2])}},
    )


def test_valid_dataset_contract() -> None:
    dataset = make_dataset()

    dataset.validate()

    assert len(dataset) == 4
    assert dataset.shape == (4, 2, 1)
    assert dataset.n_channels == 2
    assert dataset.n_features == 1


def test_invalid_data_shape_fails() -> None:
    dataset = BaseDataset(data=np.ones((4, 2)))

    with pytest.raises(PipelineError, match="shape"):
        dataset.validate()


def test_misaligned_fields_fail() -> None:
    dataset = make_dataset()
    dataset.labels = np.array([1])

    with pytest.raises(PipelineError, match="labels"):
        dataset.validate()


def test_copy_does_not_alias_arrays_or_metadata() -> None:
    dataset = make_dataset()
    copied = dataset.copy()

    copied.data[0, 0, 0] = 99
    copied.metadata["aligned_time"]["file_id"][0] = 99

    assert dataset.data[0, 0, 0] == 1
    assert dataset.metadata["aligned_time"]["file_id"][0] == 1


def test_with_split_tags_copy() -> None:
    dataset = make_dataset()

    split = dataset.with_split("train", data_tag="example")

    assert split.split == "train"
    assert split.data_tag == "example"
    assert dataset.split is None


def test_dataset_bundle_splits() -> None:
    dataset = make_dataset()
    bundle = DatasetBundle(train=dataset, val=dataset.copy(), test=dataset.copy())

    assert set(bundle.splits()) == {"train", "val", "test"}
