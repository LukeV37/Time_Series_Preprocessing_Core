# Migration Map: `src/preprocessing_core/api.py`

## Purpose

Tiny public entrypoint layer. This file gives users the simplest path into the framework and keeps construction details out of user code.

## Scope

Owns:

- `load_dataset_bundle(path)`
- `build_bundle(config)`

Does not own:

- YAML parsing internals
- pipeline scheduling
- validation implementation
- loader/transform/split logic

## Pseudocode

```python
"""Public entrypoints for building preprocessing dataset bundles."""

from pathlib import Path

from preprocessing_core.config import PipelineConfig, load_config
from preprocessing_core.datasets import DatasetBundle
from preprocessing_core.pipeline import PreprocessingPipeline


def build_bundle(config: PipelineConfig) -> DatasetBundle:
    """Build a dataset bundle from an already parsed pipeline config."""
    return PreprocessingPipeline.from_config(config).build()


def load_dataset_bundle(path: str | Path) -> DatasetBundle:
    """Load a YAML config file and build its dataset bundle."""
    return build_bundle(load_config(path))
```

## Delegation rules

- `api.py` delegates config parsing to `config.py`.
- `api.py` delegates orchestration to `PreprocessingPipeline`.
- Do not duplicate pipeline steps as a separate function here.

## Error handling

- Do not catch exceptions just to re-raise.
- Let `ConfigError` and `PipelineError` propagate.
- If you add a future CLI layer, that layer can decide how to display errors.

## Logging

No logging needed here. Pipeline logs build start/end.

## Human readability

- Keep this file short enough to read in seconds.
- If this file grows beyond public wrappers, move logic elsewhere.

## Tests

- `load_dataset_bundle(path)` calls `load_config` then builds.
- `build_bundle(config)` returns a `DatasetBundle`.
- Invalid config errors are not swallowed.
