"""Thin orchestration placeholder for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from preprocessing_core.config import PipelineConfig
from preprocessing_core.datasets import DatasetBundle


@dataclass
class PreprocessingPipeline:
    """Orchestrates one configured preprocessing run."""

    config: PipelineConfig

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "PreprocessingPipeline":
        """Create a pipeline from validated config."""
        config.validate()
        return cls(config=config)

    def build(self) -> DatasetBundle:
        """Build the train/val/test bundle."""
        raise NotImplementedError("Pipeline build flow is not implemented yet")
