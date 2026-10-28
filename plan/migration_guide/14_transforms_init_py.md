# Migration Map: `src/preprocessing_core/transforms/__init__.py`

## Purpose

Expose and register built-in generic transforms.

## Scope

Owns:

- built-in transform imports
- optional `register_builtin_transforms()` helper

Does not own:

- transform implementation
- transform scheduling

## Pseudocode

```python
"""Built-in generic transforms for preprocessing_core."""

from preprocessing_core.transforms.basic import (
    drop_channels,
    drop_features,
    fill_nan,
    interpolate_nan_per_channel,
    keep_channels,
    keep_features,
    validate_monotonic_timestamps,
)
from preprocessing_core.transforms.filtering import filter_quantile_timesteps
from preprocessing_core.transforms.scaling import robust_scale, standard_scale


def register_builtin_transforms(*, replace: bool = False) -> None:
    from preprocessing_core.registry import register_transform

    register_transform("fill_nan", fill_nan, replace=replace)
    register_transform("interpolate_nan_per_channel", interpolate_nan_per_channel, replace=replace)
    register_transform("keep_channels", keep_channels, replace=replace)
    register_transform("drop_channels", drop_channels, replace=replace)
    register_transform("keep_features", keep_features, replace=replace)
    register_transform("drop_features", drop_features, replace=replace)
    register_transform("validate_monotonic_timestamps", validate_monotonic_timestamps, replace=replace)
    register_transform("filter_quantile_timesteps", filter_quantile_timesteps, replace=replace)
    register_transform("standard_scale", standard_scale, replace=replace)
    register_transform("robust_scale", robust_scale, replace=replace)
```

## Human readability

- This file should only expose/register names.
- Keep naming identical to YAML names where possible.

## Error handling

No direct error handling.

## Logging

No logging.

## Tests

- built-in transforms register under expected names
- no optional dependencies needed
