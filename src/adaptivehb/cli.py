"""Command-line interface for the AdaptiveHb framework.

A thin, config-driven wrapper over :class:`~adaptivehb.pipeline.HbPipeline`
(Decision 030). Every pipeline mode is exposed as a subcommand so a full
experiment — or any individual mode — can be launched reproducibly from a single
command in any environment (notably the torch/GPU environment used for real
training runs)::

    adaptivehb train --config-dir configs --base-dir runs/exp1 --epochs 50
    adaptivehb experiment --name paper1 --dataset-root /data/hb --epochs 50
    python -m adaptivehb evaluate --base-dir runs/exp1

The CLI holds no framework logic of its own: it parses arguments, constructs the
facade via :meth:`HbPipeline.from_config_dir`, dispatches to the matching public
method, and prints the returned summary as JSON. Nothing is hardcoded — the
config directory, base directory, dataset root, epoch override, and experiment
name all come from arguments (with the same defaults as the facade). Heavy ML
dependencies are never imported here, so the CLI is importable and testable
without torch.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from adaptivehb.exceptions import AdaptiveHbError
from adaptivehb.logging import get_logger
from adaptivehb.version import __version__

# Subcommands that accept an ``--epochs`` override (training-like modes).
_EPOCH_COMMANDS: frozenset[str] = frozenset({"train", "resume", "experiment", "crossval"})

# Map each subcommand to the HbPipeline method that implements it. Keeping this
# as data (not branching) means new facade modes are one line to expose.
_COMMANDS: dict[str, str] = {
    "build": "build",
    "train": "train",
    "resume": "resume",
    "evaluate": "evaluate",
    "predict": "predict",
    "deploy": "deploy",
    "experiment": "experiment",
    "crossval": "cross_validate",
}


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``adaptivehb`` command.

    Returns:
        A fully configured :class:`argparse.ArgumentParser` with one subparser
        per pipeline mode.
    """
    parser = argparse.ArgumentParser(
        prog="adaptivehb",
        description="Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="command", required=True)

    descriptions = {
        "build": "Validate the framework wiring (BUILD mode).",
        "train": "Validate, split, train the models, and register them (TRAINING mode).",
        "resume": "Continue training from the latest checkpoints (RESUME mode).",
        "evaluate": "Evaluate registered models on the test split (EVALUATION mode).",
        "predict": "Run inference over held-out samples (INFERENCE mode).",
        "deploy": "Run the deployment mode (DEPLOYMENT mode).",
        "experiment": "Run and archive a full baseline-vs-adaptive experiment.",
        "crossval": "Run patient-level k-fold cross-validation and aggregate the folds.",
    }
    for command, help_text in descriptions.items():
        sub = subparsers.add_parser(command, help=help_text, description=help_text)
        _add_common_arguments(sub)
        if command in _EPOCH_COMMANDS:
            sub.add_argument(
                "--epochs",
                type=int,
                default=None,
                help="Override the number of training epochs from the config.",
            )
        if command in ("experiment", "crossval"):
            sub.add_argument(
                "--name",
                default=command,
                help="Run name (a fresh archive directory is created for it).",
            )
        if command == "crossval":
            sub.add_argument(
                "--folds",
                type=int,
                default=5,
                help="Number of cross-validation folds (k >= 2; default: 5).",
            )
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the configuration/location arguments shared by every subcommand."""
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing the framework's *.yaml config files (default: configs).",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Root directory for all framework outputs (default: current directory).",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Explicit dataset root; overrides the value in the dataset config.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the JSON result summary on stdout.",
    )


def _run_command(args: argparse.Namespace) -> dict[str, Any]:
    """Build the pipeline and dispatch the parsed subcommand.

    Args:
        args: Parsed arguments carrying ``command`` and the common options.

    Returns:
        A JSON-serializable summary of the run.
    """
    # Imported lazily so ``--help``/``--version`` never construct managers.
    from adaptivehb.pipeline import HbPipeline

    pipeline = HbPipeline.from_config_dir(
        config_dir=args.config_dir,
        base_dir=args.base_dir,
        dataset_root=args.dataset_root,
    )
    method = getattr(pipeline, _COMMANDS[args.command])
    try:
        if args.command == "crossval":
            result = method(args.name, folds=args.folds, epochs=args.epochs)
        elif args.command == "experiment":
            result = method(args.name, epochs=args.epochs)
        elif args.command in _EPOCH_COMMANDS:
            result = method(epochs=args.epochs)
        else:
            result = method()
    finally:
        pipeline.shutdown()
    return _to_summary(result)


def _to_summary(result: Any) -> dict[str, Any]:
    """Normalize a facade return value into a JSON-serializable summary."""
    if hasattr(result, "to_dict"):
        return dict(result.to_dict())
    if isinstance(result, dict):
        return result
    return {"result": result}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``adaptivehb`` command.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        A process exit code: ``0`` on success, ``1`` on a framework error, ``2``
        on a usage error (raised by argparse).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    log = get_logger("cli")
    try:
        summary = _run_command(args)
    except AdaptiveHbError as exc:
        log.error("%s failed: %s", args.command, exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(json.dumps(summary, indent=2, default=str))
    return 0


__all__ = ["build_parser", "main"]
