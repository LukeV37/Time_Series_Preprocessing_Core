import numpy as np
import pytest

from preprocessing_core import BaseDataset
from preprocessing_core.registry import (
    LOADERS,
    get_loader,
    register_loader,
)
from preprocessing_core.validation import ConfigError


def dummy_loader() -> BaseDataset:
    return BaseDataset(data=np.ones((1, 1, 1)))


def test_register_and_get_loader() -> None:
    LOADERS.clear()

    register_loader("dummy", dummy_loader)

    assert get_loader("dummy") is dummy_loader


def test_duplicate_loader_fails_unless_replace() -> None:
    LOADERS.clear()
    register_loader("dummy", dummy_loader)

    with pytest.raises(ConfigError, match="already registered"):
        register_loader("dummy", dummy_loader)

    register_loader("dummy", dummy_loader, replace=True)


def test_blank_name_and_non_callable_fail() -> None:
    LOADERS.clear()

    with pytest.raises(ConfigError, match="blank name"):
        register_loader("", dummy_loader)

    with pytest.raises(ConfigError, match="callable"):
        register_loader("bad", object())  # type: ignore[arg-type]


def test_unknown_loader_lists_available() -> None:
    LOADERS.clear()
    register_loader("dummy", dummy_loader)

    with pytest.raises(ConfigError, match="dummy"):
        get_loader("missing")
