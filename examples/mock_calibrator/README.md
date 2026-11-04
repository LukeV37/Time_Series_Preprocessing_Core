# Mock Calibrator Example

This example shows how project-specific dataset code plugs into the generic preprocessing core.

The mock calibrator format is not built into `preprocessing_core`. Instead, this example provides a small adapter layer:

- `loader.py` loads the generated mock calibrator `.npz` archive into a core `BaseDataset`.
- `register.py` registers that loader as `mock_calibrator_npz`.
- `pipeline.yaml` selects the registered loader, split policy, and transforms.
- `src/` contains the toy time-series generator and its config. See `src/README.md` for generation details.

## Quick start

From the repository root, generate the toy time-series data:

```bash
python examples/mock_calibrator/src/generate_toy_timeseries.py
```

Then run the preprocessing pipeline:

```bash
python scripts/run_experiment.py \
  --config examples/mock_calibrator/pipeline.yaml \
  --setup examples.mock_calibrator.register
```

## Registration pattern

The generic runner accepts setup modules through `--setup`:

```bash
--setup examples.mock_calibrator.register
```

The runner imports that module and calls its `register()` function. In this example, `register.py` registers the custom loader:

```python
register_loader("mock_calibrator_npz", load_mock_calibrator_npz)
```

That registration makes the loader available to `pipeline.yaml`:

```yaml
loader:
  type: mock_calibrator_npz
  params:
    path: examples/mock_calibrator/src/toy_timeseries.npz
```

This keeps experiment-specific loading code outside the core package while still allowing pipelines to remain declarative.

## Pipeline overview

`pipeline.yaml` demonstrates the intended preprocessing flow:

1. Load data with the registered `mock_calibrator_npz` loader.
2. Apply `fill_nan` before splitting.
3. Split chronologically into train/validation/test sets.
4. Apply `repair_rolling_outliers` to train and validation data.
5. Fit `standard_scale` on train data and apply it to train/validation/test.

Built-in transforms and split policies are registered by the generic runner. The mock calibrator-specific loader is registered by this example's setup module.

For details on how the synthetic dataset is generated, see `src/README.md`.
