# Migration Map: `src/preprocessing_core/config.py`

## Purpose

Parse YAML into typed config dataclasses and enforce schema-level rules before any data is loaded.

This file should make the framework easy for users: defaults are applied once here, typos are rejected early, and invalid combinations fail before expensive I/O.

## Scope

Owns:

- YAML loading with `yaml.safe_load`
- dataclasses for config objects
- default application
- strict top-level schema validation
- pre/post split transform config validation

Does not own:

- callable registry lookup
- runtime dataset validation
- transform execution
- split index validation

## Config dataclasses

```python
"""YAML configuration parsing for preprocessing pipelines."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from preprocessing_core.validation import ConfigError

SplitName = Literal["train", "val", "test"]
VALID_SPLITS = {"train", "val", "test"}


@dataclass
class LoaderConfig:
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.type or not isinstance(self.type, str):
            raise ConfigError("loader.type must be a non-empty string")
        if not isinstance(self.params, dict):
            raise ConfigError("loader.params must be a mapping")


@dataclass
class TransformConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    apply: tuple[str, ...] | None = None
    fit: str | None = None
    state_key: str | None = None

    def validate_for_stage(self, stage: str) -> None:
        if not self.name:
            raise ConfigError(f"{stage} transform is missing name")
        if stage == "pre_split":
            if self.apply is not None:
                raise ConfigError(f"pre_split transform {self.name!r} cannot use apply")
            if self.fit is not None:
                raise ConfigError(f"pre_split transform {self.name!r} cannot use fit")
        if stage == "post_split":
            apply = self.apply or ("train", "val", "test")
            validate_apply_and_fit(stage=stage, name=self.name, apply=apply, fit=self.fit)


@dataclass
class SplitConfig:
    policy: str
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.policy:
            raise ConfigError("split.policy must be a non-empty string")
        if not isinstance(self.params, dict):
            raise ConfigError("split.params must be a mapping")


@dataclass
class PipelineConfig:
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
        self.loader.validate()
        self.split.validate()
        for step in self.pre_split:
            step.validate_for_stage("pre_split")
        for step in self.post_split:
            step.validate_for_stage("post_split")
```

## Parsing pseudocode

```python
TOP_LEVEL_KEYS = {"version", "loader", "pre_split", "split", "post_split", "metadata"}
TRANSFORM_KEYS = {"name", "params", "apply", "fit", "state_key"}


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config {path} must contain a mapping at top level")

    config = parse_config_dict(raw, config_path=str(path))
    config.validate()
    return config


def parse_config_dict(raw: dict[str, Any], *, config_path: str | None = None) -> PipelineConfig:
    unknown = set(raw) - TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"Unknown top-level config keys: {sorted(unknown)}")

    if "loader" not in raw:
        raise ConfigError("Missing required top-level key 'loader'")
    if "split" not in raw:
        raise ConfigError("Missing required top-level key 'split'")

    return PipelineConfig(
        version=raw.get("version"),
        loader=parse_loader_config(raw["loader"]),
        split=parse_split_config(raw["split"]),
        pre_split=parse_transform_list(raw.get("pre_split", []), stage="pre_split"),
        post_split=parse_transform_list(raw.get("post_split", []), stage="post_split"),
        metadata=dict(raw.get("metadata") or {}),
        config_path=config_path,
    )
```

## Defaults to apply here

- `pre_split`: `[]`
- `post_split`: `[]`
- `metadata`: `{}`
- transform `params`: `{}`
- post-split `apply`: all splits, but this can be applied later in scheduler if kept as `None`
- `state_key`: derived by scheduler, not config parser

## Error handling

Use `ConfigError` for:

- invalid YAML shape
- unknown top-level key
- unknown transform field
- bad `apply` / `fit`
- missing required sections
- invalid schema version

Let `FileNotFoundError` stay as `FileNotFoundError` unless wrapping adds useful context.

## Logging

No logging required. Config parsing should be deterministic and quiet.

## Human readability

- Keep parsing functions small.
- Do not mix runtime registry lookup into config parsing.
- Include exact field names in error messages.

## Tests

- valid minimal config parses
- unknown top-level key fails
- unknown transform key fails
- missing loader/split fails
- `pre_split` with `apply` or `fit` fails
- post-split `fit: val` fails
- post-split `fit: train` not in `apply` fails
- defaults are applied consistently
