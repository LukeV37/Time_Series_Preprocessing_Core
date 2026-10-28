# Preprocessing Core Migration Guide

This directory contains implementation guidance for `plan/MINIMAL_PREPROCESSING_CORE_PLAN.md`.

It is written for future agents implementing the public `preprocessing_core` mini-framework. The goal is to preserve the framework ambition while keeping the code clean, readable, testable, and aligned with `NEW_REPO_STYLE_GUIDE.md`.

## Main conclusion

The plan is directionally good. It does **not** need to become less framework-like. Since the target is a public preprocessing framework, it is appropriate for the core to provide rules, scheduling, validation, registries, and reusable contracts.

The main implementation risk is not the existence of a framework. The risk is putting too much framework behavior inside `PreprocessingPipeline` or any single config/validation module.

## Source of truth

`MINIMAL_PREPROCESSING_CORE_PLAN.md` is the authoritative design for this work.
This directory provides implementation guidance for that plan.
If guidance here appears to conflict with the plan, follow the plan and update the guide.

## File-by-file migration maps

| Planned file | Guide |
| --- | --- |
| `src/preprocessing_core/__init__.py` | [`01_init_py.md`](01_init_py.md) |
| `src/preprocessing_core/api.py` | [`02_api_py.md`](02_api_py.md) |
| `src/preprocessing_core/datasets.py` | [`03_datasets_py.md`](03_datasets_py.md) |
| `src/preprocessing_core/config.py` | [`04_config_py.md`](04_config_py.md) |
| `src/preprocessing_core/pipeline.py` | [`05_pipeline_py.md`](05_pipeline_py.md) |
| `src/preprocessing_core/registry.py` | [`06_registry_py.md`](06_registry_py.md) |
| `src/preprocessing_core/validation.py` | [`07_validation_py.md`](07_validation_py.md) |
| `src/preprocessing_core/selection.py` | [`08_selection_py.md`](08_selection_py.md) |
| `src/preprocessing_core/loaders/__init__.py` | [`09_loaders_init_py.md`](09_loaders_init_py.md) |
| `src/preprocessing_core/loaders/npy.py` | [`10_loaders_npy_py.md`](10_loaders_npy_py.md) |
| `src/preprocessing_core/loaders/npz.py` | [`11_loaders_npz_py.md`](11_loaders_npz_py.md) |
| `src/preprocessing_core/loaders/csv.py` | [`12_loaders_csv_py.md`](12_loaders_csv_py.md) |
| `src/preprocessing_core/loaders/hdf5.py` | [`13_loaders_hdf5_py.md`](13_loaders_hdf5_py.md) |
| `src/preprocessing_core/transforms/__init__.py` | [`14_transforms_init_py.md`](14_transforms_init_py.md) |
| `src/preprocessing_core/transforms/basic.py` | [`15_transforms_basic_py.md`](15_transforms_basic_py.md) |
| `src/preprocessing_core/transforms/filtering.py` | [`16_transforms_filtering_py.md`](16_transforms_filtering_py.md) |
| `src/preprocessing_core/transforms/scaling.py` | [`17_transforms_scaling_py.md`](17_transforms_scaling_py.md) |
| `src/preprocessing_core/splits/__init__.py` | [`18_splits_init_py.md`](18_splits_init_py.md) |
| `src/preprocessing_core/splits/chronological.py` | [`19_splits_chronological_py.md`](19_splits_chronological_py.md) |
| `src/preprocessing_core/splits/ranges.py` | [`20_splits_ranges_py.md`](20_splits_ranges_py.md) |
| `src/preprocessing_core/splits/seasonal.py` | [`22_splits_seasonal_py.md`](22_splits_seasonal_py.md) |
| Tests and docs | [`21_tests_and_docs.md`](21_tests_and_docs.md) |

## Explicit adoption boundary

This migration guide is for implementing the standalone `preprocessing_core` package. It is **not** a promise that the current `CrossExperimentalAIDQM/src/anldq/` trainer, infer, analysis, baselines, and CLI modules can switch to it without follow-on migration work.

In particular, current shipped code still contains legacy assumptions that are intentionally outside the first implementation scope of this guide, including:

- tuple-style dataset loading instead of `DatasetBundle`;
- legacy dataset field names like `meta` and `headers`;
- direct `channel_groups` attribute access;
- GM2-specific class-name checks;
- downstream reliance on legacy dataset-side score aggregation config, `raw_data`, and older split/group metadata conventions.

Treat those as a separate adoption track after the standalone core exists and passes its own tests.

## Framework design rules to preserve

1. `PreprocessingPipeline` sequences work; helper functions make decisions.
2. `BaseDataset` carries data; it does not load, split, transform, or evaluate.
3. YAML selects registered names; Python supplies behavior.
4. Public transforms operate on `BaseDataset`, not raw arrays.
5. Every row/channel selection goes through `selection.py` helpers.
6. Config errors fail at load time where possible.
7. Runtime errors include stage, transform name, and split name where relevant.
8. Logging is concise and stage-level, never noisy inner-loop logging.
9. Public core stays generic; experiment-specific logic lives in private plugins.
10. The mini-framework should reduce downstream decisions, not hide dangerous behavior.
