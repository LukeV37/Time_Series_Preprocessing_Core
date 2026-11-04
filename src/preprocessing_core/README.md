# preprocessing_core

Core package for building YAML-configured preprocessing pipelines for time-series datasets.

The package is intentionally small. It provides generic dataset containers, config parsing, registries, split policies, transforms, and pipeline assembly. Project-specific dataset loading should live outside this package and be registered by examples or downstream projects.

## Layout

```text
api.py          Public convenience functions for loading configs and building bundles.
config.py       YAML loading and conversion into typed pipeline config objects.
datasets.py     Shared dataset containers such as `BaseDataset` and `DatasetBundle`.
pipeline.py     Main pipeline assembly logic.
registry.py     Runtime registries for loaders, split policies, and transforms.
selection.py    Helpers for selecting which splits a transform applies to.
validation.py   Config and pipeline error types.

splits/         Built-in split policies.
transforms/     Built-in preprocessing transforms.
```

## Built-in extension points

The core uses registries so that YAML configs can refer to operations by name:

- loaders are registered with `register_loader(...)`,
- split policies are registered with `register_split_policy(...)`,
- transforms are registered with `register_transform(...)`.

Built-in split policies and transforms are registered by the generic runner. Project-specific loaders are expected to be registered by setup modules, such as `examples.mock_calibrator.register`.

## Typical flow

1. Load a YAML config.
2. Resolve the configured loader, split policy, and transforms from registries.
3. Load a `BaseDataset`.
4. Apply pre-split transforms.
5. Split into a `DatasetBundle`.
6. Apply post-split transforms.
