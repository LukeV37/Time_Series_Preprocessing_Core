"""Register toy calibrator operations with preprocessing_core."""

from __future__ import annotations

from preprocessing_core import register_loader

from examples.toy_calibrator.loader import load_toy_npz


def register() -> None:
    """Register custom operations needed by the toy calibrator example."""
    register_loader("toy_npz", load_toy_npz)
