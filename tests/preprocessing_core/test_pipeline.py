import numpy as np
import pytest

from preprocessing_core.config import parse_config_dict
from preprocessing_core.datasets import BaseDataset
from preprocessing_core.pipeline import PreprocessingPipeline
from preprocessing_core.registry import (
    LOADERS,
    SPLIT_POLICIES,
    TRANSFORMS,
    register_loader,
    register_split_policy,
    register_transform,
)
from preprocessing_core.transforms import fill_nan, standard_scale
from preprocessing_core.validation import PipelineError


def setup_function() -> None:
    LOADERS.clear()
    SPLIT_POLICIES.clear()
    TRANSFORMS.clear()


def make_config(*, pre_split=None, post_split=None):
    return parse_config_dict(
        {
            "version": 1,
            "loader": {"type": "dummy"},
            "pre_split": pre_split or [],
            "split": {"policy": "thirds"},
            "post_split": post_split or [],
            "metadata": {"dataset_name": "unit_test"},
        }
    )


def dummy_loader() -> BaseDataset:
    data = np.arange(12, dtype=float).reshape(6, 1, 2)
    data[1, 0, 0] = np.nan
    return BaseDataset(data=data)


def thirds_split(dataset: BaseDataset) -> dict[str, np.ndarray]:
    return {
        "train": np.array([0, 1]),
        "val": np.array([2, 3]),
        "test": np.array([4, 5]),
    }


def add_constant(
    dataset: BaseDataset,
    *,
    state: dict[str, object],
    fit: bool,
    value: float,
) -> BaseDataset:
    _ = state, fit
    result = dataset.copy()
    result.data = result.data + value
    return result


def bad_transform(dataset: BaseDataset, *, state: dict[str, object], fit: bool):
    _ = dataset, state, fit
    return "not a dataset"


def register_core_test_parts() -> None:
    register_loader("dummy", dummy_loader)
    register_split_policy("thirds", thirds_split)


def test_pre_split_transform_runs_before_split() -> None:
    register_core_test_parts()
    register_transform("fill_nan", fill_nan)

    config = make_config(pre_split=[{"name": "fill_nan", "params": {"value": -1.0}}])
    bundle = PreprocessingPipeline.from_config(config).build()

    assert bundle.train.data[1, 0, 0] == -1.0
    assert bundle.train.preprocess_steps == ["pre_split:0:fill_nan"]


def test_post_split_apply_skips_test() -> None:
    register_core_test_parts()
    register_transform("add_constant", add_constant)

    config = make_config(
        post_split=[
            {"name": "add_constant", "apply": ["train", "val"], "params": {"value": 100.0}}
        ]
    )
    bundle = PreprocessingPipeline.from_config(config).build()

    assert bundle.train.data[0, 0, 1] == 101.0
    assert bundle.val.data[0, 0, 0] == 104.0
    assert bundle.test.data[0, 0, 0] == 8.0
    assert bundle.test.preprocess_steps == []


def test_standard_scale_fitted_state_is_saved_in_bundle_metadata() -> None:
    register_core_test_parts()
    register_transform("fill_nan", fill_nan)
    register_transform("standard_scale", standard_scale)

    config = make_config(
        pre_split=[{"name": "fill_nan"}],
        post_split=[
            {
                "name": "standard_scale",
                "fit": "train",
                "apply": ["train", "val", "test"],
                "state_key": "scale_state",
            }
        ],
    )
    bundle = PreprocessingPipeline.from_config(config).build()

    state = bundle.metadata["fitted_state"]["scale_state"]
    assert state["mean"].shape == (1, 2)
    assert state["std"].shape == (1, 2)
    np.testing.assert_allclose(np.mean(bundle.train.data, axis=0), np.zeros((1, 2)))
    assert bundle.val.preprocess_steps[-1] == "post_split:0:standard_scale"


def test_transform_returning_wrong_type_fails() -> None:
    register_core_test_parts()
    register_transform("bad", bad_transform)

    config = make_config(pre_split=[{"name": "bad"}])

    with pytest.raises(PipelineError, match="did not return BaseDataset"):
        PreprocessingPipeline.from_config(config).build()
