# Preprocessing-Core

A minimal, standalone preprocessing core for YAML-configured time-series datasets.

The package separates generic preprocessing mechanics from project-specific dataset code. Experiments can register custom loaders, choose built-in split policies and transforms in YAML, and build a `DatasetBundle` through a generic runner.

## Quick example

The mock calibrator example shows the current intended workflow: experiment code registers a custom loader, YAML selects that registered loader plus built-in transforms and split policy, and the generic runner builds a `DatasetBundle`.

```bash
python examples/mock_calibrator/src/generate_toy_timeseries.py

python scripts/run_experiment.py \
  --config examples/mock_calibrator/pipeline.yaml \
  --setup examples.mock_calibrator.register
```

See `examples/mock_calibrator/README.md` for the full example notes.

## Repository layout

```text
src/preprocessing_core/   Core package: config loading, registries, splits, transforms, and pipeline assembly.
scripts/                  Small command-line entry points for running configured preprocessing jobs.
examples/mock_calibrator/ Example showing how project-specific dataset code plugs into the core.
tests/                    Unit tests for core behavior and built-in operations.
pyproject.toml            Package metadata, dependencies, pytest config, and lint settings.
```

## Core idea

A preprocessing job is defined by:

1. a YAML config,
2. registered loader, split, and transform functions,
3. the generic pipeline builder.

Project-specific code stays outside `preprocessing_core`; it only needs to register adapters that the YAML config can reference.
