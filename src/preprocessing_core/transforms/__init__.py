"""Built-in transforms for preprocessing_core."""

from preprocessing_core.transforms.fill import fill_nan
from preprocessing_core.transforms.rolling import repair_rolling_outliers
from preprocessing_core.transforms.scale import standard_scale


def register_builtin_transforms(*, replace: bool = False) -> None:
    """Register built-in transforms with the core registry."""
    from preprocessing_core.registry import register_transform

    register_transform("fill_nan", fill_nan, replace=replace)
    register_transform("repair_rolling_outliers", repair_rolling_outliers, replace=replace)
    register_transform("standard_scale", standard_scale, replace=replace)


__all__ = [
    "fill_nan",
    "register_builtin_transforms",
    "repair_rolling_outliers",
    "standard_scale",
]
