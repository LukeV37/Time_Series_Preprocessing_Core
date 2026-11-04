# Scripts

Small command-line entry points for running preprocessing workflows.

## `run_experiment.py`

Runs a YAML-configured preprocessing pipeline.

The script:

1. adds an import root to Python imports,
2. registers built-in split policies and transforms,
3. imports optional setup modules,
4. builds a `DatasetBundle`,
5. prints a short split summary.

Example:

```bash
python scripts/run_experiment.py \
  --config examples/mock_calibrator/pipeline.yaml \
  --setup examples.mock_calibrator.register
```

Useful arguments:

- `--config`: path to the pipeline YAML file.
- `--setup`: import path to a module with a callable `register()` function. Can be passed more than once.
- `--import-root`: directory added to Python imports before setup modules load. Defaults to the repository root.
- `--replace`: allow built-in registrations to replace existing registry names.
