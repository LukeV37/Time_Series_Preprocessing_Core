"""Small public entry points for building preprocessing bundles."""

from __future__ import annotations

from pathlib import Path

from preprocessing_core.config import PipelineConfig, load_config
from preprocessing_core.datasets import DatasetBundle


def build_bundle(config: PipelineConfig) -> DatasetBundle:
    """Build a dataset bundle from a parsed pipeline config."""
    from preprocessing_core.pipeline import PreprocessingPipeline

    return PreprocessingPipeline.from_config(config).build()


def load_dataset_bundle(path: str | Path) -> DatasetBundle:
    """Load a config file and build its dataset bundle."""
    return build_bundle(load_config(path))
