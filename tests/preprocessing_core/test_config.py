import pytest

from preprocessing_core.config import parse_config_dict
from preprocessing_core.validation import ConfigError


def minimal_config() -> dict:
    return {
        "version": 1,
        "loader": {"type": "npz", "params": {"path": "data.npz"}},
        "split": {"policy": "chronological", "params": {"ratios": [0.6, 0.2, 0.2]}},
    }


def test_minimal_config_parses_with_defaults() -> None:
    config = parse_config_dict(minimal_config())

    assert config.version == 1
    assert config.pre_split == []
    assert config.post_split == []
    assert config.metadata == {}


def test_unknown_top_level_key_fails() -> None:
    raw = minimal_config()
    raw["typo"] = True

    with pytest.raises(ConfigError, match="Unknown top-level"):
        parse_config_dict(raw)


def test_unknown_transform_key_fails() -> None:
    raw = minimal_config()
    raw["post_split"] = [{"name": "fill_nan", "apl": ["train"]}]

    with pytest.raises(ConfigError, match="Unknown post_split transform keys"):
        parse_config_dict(raw)


def test_missing_loader_fails() -> None:
    raw = minimal_config()
    raw.pop("loader")

    with pytest.raises(ConfigError, match="loader"):
        parse_config_dict(raw)


def test_pre_split_rejects_apply_and_fit() -> None:
    raw = minimal_config()
    raw["pre_split"] = [{"name": "fill_nan", "apply": ["train"]}]

    with pytest.raises(ConfigError, match="cannot use apply"):
        parse_config_dict(raw)

    raw["pre_split"] = [{"name": "fill_nan", "fit": "train"}]
    with pytest.raises(ConfigError, match="cannot use fit"):
        parse_config_dict(raw)


def test_post_split_fit_must_be_train_and_applied() -> None:
    raw = minimal_config()
    raw["post_split"] = [{"name": "scale", "fit": "val", "apply": ["val"]}]

    with pytest.raises(ConfigError, match="fit must be 'train'"):
        parse_config_dict(raw)

    raw["post_split"] = [{"name": "scale", "fit": "train", "apply": ["val"]}]
    with pytest.raises(ConfigError, match="does not apply to train"):
        parse_config_dict(raw)
