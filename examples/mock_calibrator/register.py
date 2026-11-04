"""Register mock calibrator operations with preprocessing_core."""

from __future__ import annotations

from examples.mock_calibrator.loader import load_mock_calibrator_npz

from preprocessing_core import register_loader


def register() -> None:
    """Register custom operations needed by the mock calibrator example."""
    register_loader("mock_calibrator_npz", load_mock_calibrator_npz)
