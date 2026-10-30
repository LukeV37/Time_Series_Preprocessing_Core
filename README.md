# preprocessing-core

A minimal, standalone preprocessing core for YAML-configured time-series datasets.

The authoritative design lives in `plan/MINIMAL_PREPROCESSING_CORE_PLAN.md` while this package is being bootstrapped.

## Quick example

The toy calibrator example shows the current intended workflow: experiment code registers a custom loader, YAML selects that registered loader plus built-in transforms and split policy, and the generic runner builds a `DatasetBundle`.

```bash
python examples/toy_calibrator/generate_toy_timeseries.py

python scripts/run_experiment.py \
  --config examples/toy_calibrator/pipeline.yaml \
  --setup examples.toy_calibrator.register
```

See `examples/toy_calibrator/README.md` for the full example notes.
