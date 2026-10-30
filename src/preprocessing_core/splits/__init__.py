"""Built-in split policies for preprocessing_core."""

from preprocessing_core.splits.chronological import chronological_split


def register_builtin_split_policies(*, replace: bool = False) -> None:
    """Register built-in split policies with the core registry."""
    from preprocessing_core.registry import register_split_policy

    register_split_policy("chronological", chronological_split, replace=replace)


__all__ = ["chronological_split", "register_builtin_split_policies"]
