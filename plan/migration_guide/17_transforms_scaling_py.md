# Migration Map: `src/preprocessing_core/transforms/scaling.py`

## Purpose

Train-fitted, serializable, NumPy-only scaling transforms.

Scaling is not a special pipeline stage. It is a normal post-split transform using `fit: train` and reusable state.

## Scope

Owns:

- `standard_scale`
- `robust_scale`
- NumPy statistic calculation
- state format validation for scaling transforms

Does not own:

- sklearn estimators
- arbitrary scaler objects
- persistence to disk beyond returning serializable state in `state`

## Shared rules

- If `fit=True`, compute stats and write them into `state`.
- If `fit=False`, require stats already exist in `state`.
- Never compute fallback val/test stats silently.
- Store plain arrays/scalars only.
- Return a valid `BaseDataset`.

## Supported modes

Start with one clear mode:

- `per_channel_feature`: stats shape `(C, F)`, computed across time axis.

Only add other modes with tests and docs.

## `standard_scale` pseudocode

```python
"""Train-fitted scaling transforms."""

import numpy as np

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.validation import ConfigError, PipelineError


def standard_scale(
    dataset: BaseDataset,
    *,
    state: dict[str, object],
    fit: bool,
    mode: str = "per_channel_feature",
    epsilon: float = 1e-8,
) -> BaseDataset:
    """Scale data using train-fitted mean and standard deviation."""
    if mode != "per_channel_feature":
        raise ConfigError(f"Unsupported standard_scale mode {mode!r}")

    if fit:
        mean = np.nanmean(dataset.data, axis=0)
        std = np.nanstd(dataset.data, axis=0)
        std = np.where(std < epsilon, 1.0, std)
        state["mean"] = mean
        state["std"] = std
        state["mode"] = mode
    else:
        _require_state_keys(state, ["mean", "std", "mode"], transform="standard_scale")
        if state["mode"] != mode:
            raise PipelineError("standard_scale state mode does not match configured mode")
        mean = np.asarray(state["mean"])
        std = np.asarray(state["std"])

    result = dataset.copy()
    result.data = (result.data - mean) / std
    result.validate()
    return result
```

## `robust_scale` pseudocode

```python
def robust_scale(
    dataset: BaseDataset,
    *,
    state: dict[str, object],
    fit: bool,
    mode: str = "per_channel_feature",
    q_low: float = 25.0,
    q_high: float = 75.0,
    epsilon: float = 1e-8,
) -> BaseDataset:
    """Scale data using train-fitted median and interquantile range."""
    if mode != "per_channel_feature":
        raise ConfigError(f"Unsupported robust_scale mode {mode!r}")
    if not 0 <= q_low < q_high <= 100:
        raise ConfigError("robust_scale requires 0 <= q_low < q_high <= 100")

    if fit:
        center = np.nanmedian(dataset.data, axis=0)
        low = np.nanpercentile(dataset.data, q_low, axis=0)
        high = np.nanpercentile(dataset.data, q_high, axis=0)
        scale = np.where((high - low) < epsilon, 1.0, high - low)
        state.update({"center": center, "scale": scale, "mode": mode, "q_low": q_low, "q_high": q_high})
    else:
        _require_state_keys(state, ["center", "scale", "mode"], transform="robust_scale")
        center = np.asarray(state["center"])
        scale = np.asarray(state["scale"])

    result = dataset.copy()
    result.data = (result.data - center) / scale
    result.validate()
    return result
```

## Error handling

- Missing state on val/test: `PipelineError` naming transform and required keys.
- Unsupported mode: `ConfigError`.
- Non-finite output policy should be explicit. Prefer to let validation/transforms catch separately unless scaling itself produces invalid values.

## Logging

No logging. Pipeline logs fitted-state key creation/reuse at DEBUG if desired.

## Human readability

- Keep state keys obvious: `mean`, `std`, `center`, `scale`.
- Do not store estimator objects.
- Do not add many modes without docs and tests.

## Tests

- train computes state
- val/test use train state
- val/test fail if state missing
- zero std/IQR handled with epsilon rule
- state is serializable
- scaling preserves metadata and labels
