"""Run a preprocessing experiment from a config and setup modules."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from preprocessing_core import load_dataset_bundle
from preprocessing_core.splits import register_builtin_split_policies
from preprocessing_core.transforms import register_builtin_transforms


def main() -> None:
    """Register operations, build the configured bundle, and print a summary."""
    args = parse_args()
    add_import_root(args.import_root)

    register_builtin_split_policies(replace=args.replace)
    register_builtin_transforms(replace=args.replace)
    for setup_module in args.setup:
        register_setup_module(setup_module)

    bundle = load_dataset_bundle(args.config)
    print_bundle_summary(bundle)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a preprocessing experiment.")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config.")
    parser.add_argument(
        "--setup",
        action="append",
        default=[],
        help="Import path to a module with register(). Can be passed more than once.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Allow built-in split/transform registration to replace existing registry names.",
    )
    parser.add_argument(
        "--import-root",
        default=".",
        help="Directory added to Python imports before setup modules load.",
    )
    return parser.parse_args()


def add_import_root(path: str) -> None:
    """Add the experiment import root so setup modules can be imported."""
    import_root = str(Path(path).resolve())
    if import_root not in sys.path:
        sys.path.insert(0, import_root)


def register_setup_module(module_path: str) -> None:
    """Import one setup module and call its register() function."""
    module = importlib.import_module(module_path)
    register = getattr(module, "register", None)

    if register is None or not callable(register):
        raise RuntimeError(f"Setup module {module_path!r} must define callable register()")

    register()


def print_bundle_summary(bundle: Any) -> None:
    """Print split shapes for a built dataset bundle."""
    print(f"train: {bundle.train.shape}")
    print(f"val:   {bundle.val.shape}")
    print(f"test:  {bundle.test.shape}")


if __name__ == "__main__":
    main()
