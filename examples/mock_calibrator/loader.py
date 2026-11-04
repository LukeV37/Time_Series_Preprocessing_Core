"""Mock calibrator loader used by the example pipeline.

This module is intentionally outside `preprocessing_core`. It demonstrates how
an experiment or private package supplies its own loader and registers it with
the core framework.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from preprocessing_core.datasets import BaseDataset
from preprocessing_core.validation import PipelineError


def load_mock_calibrator_npz(path: str | Path) -> BaseDataset:
    """Load the generated mock calibrator `.npz` archive.

    The toy archive is expected to contain:

    - `data`: required `(T, C, F)` array
    - `timestamps`: optional `(T,)` array
    - `feature_names`: optional one-dimensional array of feature names

    Other arrays in the archive are ignored by this starter loader.
    """
    path = Path(path)

    with np.load(path, allow_pickle=False) as archive:
        if "data" not in archive:
            raise PipelineError(f"Toy NPZ file {path} is missing required key 'data'")

        dataset = BaseDataset(
            data=archive["data"],
            timestamps=archive["timestamps"] if "timestamps" in archive else None,
            feature_names=_load_feature_names(archive),
            metadata={"source_format": "mock_calibrator_npz", "source_path": str(path)},
        )

    dataset.validate()
    return dataset


def _load_feature_names(archive: np.lib.npyio.NpzFile) -> list[str] | None:
    """Return toy feature names if present."""
    if "feature_names" not in archive:
        return None

    return [str(value) for value in archive["feature_names"].tolist()]
