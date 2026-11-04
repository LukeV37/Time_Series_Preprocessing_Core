"""YAML configuration parsing for preprocessing pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from preprocessing_core.validation import ConfigError, validate_apply_and_fit

VALID_SPLITS = ("train", "val", "test")
TOP_LEVEL_KEYS = {"version", "loader", "pre_split", "split", "post_split", "metadata"}
LOADER_KEYS = {"type", "params"}
SPLIT_KEYS = {"policy", "params"}
TRANSFORM_KEYS = {"name", "params", "apply", "fit", "state_key"}


@dataclass
class LoaderConfig:
    """Configured loader name and keyword parameters."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.type, str) or not self.type.strip():
            raise ConfigError("loader.type must be a non-empty string")
        if not isinstance(self.params, dict):
            raise ConfigError("loader.params must be a mapping")


@dataclass
class TransformConfig:
    """Configured preprocessing transform step."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    apply: tuple[str, ...] | None = None
    fit: str | None = None
    state_key: str | None = None

    def validate_for_stage(self, stage: str) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError(f"{stage} transform is missing name")
        if not isinstance(self.params, dict):
            raise ConfigError(f"{stage} transform {self.name!r} params must be a mapping")
        if stage == "pre_split":
            if self.apply is not None:
                raise ConfigError(f"pre_split transform {self.name!r} cannot use apply")
            if self.fit is not None:
                raise ConfigError(f"pre_split transform {self.name!r} cannot use fit")
        if stage == "post_split":
            validate_apply_and_fit(
                stage=stage,
                name=self.name,
                apply=self.apply or VALID_SPLITS,
                fit=self.fit,
            )


@dataclass
class SplitConfig:
    """Configured split policy name and keyword parameters."""

    policy: str
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.policy, str) or not self.policy.strip():
            raise ConfigError("split.policy must be a non-empty string")
        if not isinstance(self.params, dict):
            raise ConfigError("split.params must be a mapping")


@dataclass
class PipelineConfig:
    """Validated configuration for one preprocessing pipeline.

    This dataclass is the typed representation of a parsed YAML
    preprocessing config. It records the loader, split policy, optional
    transform stages, and user metadata consumed by the preprocessing
    pipeline.

    Attributes
    ----------
    version : int
        Configuration schema version. Currently only ``1`` is supported.
    loader : LoaderConfig
        Loader name and keyword parameters used to construct the full
        dataset.
    split : SplitConfig
        Split policy name and keyword parameters used to produce the
        ``train``, ``val``, and ``test`` indices.
    pre_split : list[TransformConfig]
        Transform steps applied to the full dataset before split indices are
        computed.
    post_split : list[TransformConfig]
        Transform steps applied after split datasets are materialized.
    metadata : dict[str, Any]
        User-provided metadata copied into the final DatasetBundle.
    config_path : str | None
        Source path for the parsed config, when available. The default is
        ``None``.

    Raises
    ------
    ConfigError
        Raised by validate() when the schema version, metadata, loader, split
        policy, or transform-stage constraints are invalid.
    """

    version: int
    loader: LoaderConfig
    split: SplitConfig
    pre_split: list[TransformConfig] = field(default_factory=list)
    post_split: list[TransformConfig] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    config_path: str | None = None

    def validate(self) -> None:
        if self.version != 1:
            raise ConfigError(f"Unsupported config version {self.version!r}; expected 1")
        if not isinstance(self.metadata, dict):
            raise ConfigError("metadata must be a mapping")
        self.loader.validate()
        self.split.validate()
        for step in self.pre_split:
            step.validate_for_stage("pre_split")
        for step in self.post_split:
            step.validate_for_stage("post_split")


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML preprocessing config."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config {config_path} must contain a mapping at top level")

    return parse_config_dict(raw, config_path=str(config_path))


def parse_config_dict(raw: dict[str, Any], *, config_path: str | None = None) -> PipelineConfig:
    """Parse a raw config mapping into typed config objects."""
    if not isinstance(raw, dict):
        raise ConfigError("Config must be a mapping")

    unknown = set(raw) - TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"Unknown top-level config keys: {sorted(unknown)}")
    if "version" not in raw:
        raise ConfigError("Missing required top-level key 'version'")
    if "loader" not in raw:
        raise ConfigError("Missing required top-level key 'loader'")
    if "split" not in raw:
        raise ConfigError("Missing required top-level key 'split'")

    config = PipelineConfig(
        version=raw["version"],
        loader=_parse_loader_config(raw["loader"]),
        split=_parse_split_config(raw["split"]),
        pre_split=_parse_transform_list(raw.get("pre_split", []), stage="pre_split"),
        post_split=_parse_transform_list(raw.get("post_split", []), stage="post_split"),
        metadata=dict(raw.get("metadata") or {}),
        config_path=config_path,
    )
    config.validate()
    return config


def _parse_loader_config(raw: Any) -> LoaderConfig:
    if not isinstance(raw, dict):
        raise ConfigError("loader must be a mapping")
    unknown = set(raw) - LOADER_KEYS
    if unknown:
        raise ConfigError(f"Unknown loader keys: {sorted(unknown)}")
    if "type" not in raw:
        raise ConfigError("loader.type is required")
    return LoaderConfig(type=raw["type"], params=dict(raw.get("params") or {}))


def _parse_split_config(raw: Any) -> SplitConfig:
    if not isinstance(raw, dict):
        raise ConfigError("split must be a mapping")
    unknown = set(raw) - SPLIT_KEYS
    if unknown:
        raise ConfigError(f"Unknown split keys: {sorted(unknown)}")
    if "policy" not in raw:
        raise ConfigError("split.policy is required")
    return SplitConfig(policy=raw["policy"], params=dict(raw.get("params") or {}))


def _parse_transform_list(raw: Any, *, stage: str) -> list[TransformConfig]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ConfigError(f"{stage} must be a list")
    return [_parse_transform_config(item, stage=stage) for item in raw]


def _parse_transform_config(raw: Any, *, stage: str) -> TransformConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"{stage} transform entries must be mappings")
    unknown = set(raw) - TRANSFORM_KEYS
    if unknown:
        raise ConfigError(f"Unknown {stage} transform keys: {sorted(unknown)}")
    if "name" not in raw:
        raise ConfigError(f"{stage} transform is missing name")

    apply = raw.get("apply")
    if apply is not None:
        if not isinstance(apply, list):
            raise ConfigError(f"{stage} transform {raw['name']!r} apply must be a list")
        apply = tuple(apply)

    transform = TransformConfig(
        name=raw["name"],
        params=dict(raw.get("params") or {}),
        apply=apply,
        fit=raw.get("fit"),
        state_key=raw.get("state_key"),
    )
    transform.validate_for_stage(stage)
    return transform
