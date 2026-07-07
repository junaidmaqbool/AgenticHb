"""Tests for the ``adaptivehb`` command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptivehb import cli
from adaptivehb.dataset import generate_synthetic_dataset


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

def test_parser_exposes_all_modes() -> None:
    parser = cli.build_parser()
    # Every facade command is reachable as a subcommand.
    for command in ("build", "train", "resume", "evaluate", "predict", "deploy", "experiment"):
        args = parser.parse_args([command])
        assert args.command == command


def test_parser_requires_a_command() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_epoch_flag_only_on_training_commands() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["train", "--epochs", "7"]).epochs == 7
    # evaluate has no --epochs option.
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "--epochs", "7"])


def test_experiment_name_defaults_and_overrides() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["experiment"]).name == "experiment"
    assert parser.parse_args(["experiment", "--name", "paper1"]).name == "paper1"


def test_version_flag_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert "adaptivehb" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Dispatch (end-to-end on reference models, no torch required)
# --------------------------------------------------------------------------- #

@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    generate_synthetic_dataset(root, num_patients=8, seed=3)
    return root


def _common(configs_dir: Path, base_dir: Path, dataset_root: Path) -> list[str]:
    return [
        "--config-dir", str(configs_dir),
        "--base-dir", str(base_dir),
        "--dataset-root", str(dataset_root),
    ]


def test_main_train_prints_json_summary(
    configs_dir: Path, tmp_path: Path, dataset_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["train", *_common(configs_dir, tmp_path, dataset_root), "--epochs", "2"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "training"


def test_main_quiet_suppresses_output(
    configs_dir: Path, tmp_path: Path, dataset_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["build", *_common(configs_dir, tmp_path, dataset_root), "--quiet"])
    assert code == 0
    assert capsys.readouterr().out == ""


def test_main_experiment_returns_serialized_result(
    configs_dir: Path, tmp_path: Path, dataset_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main([
        "experiment", "--name", "clitest",
        *_common(configs_dir, tmp_path, dataset_root), "--epochs", "2",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    # ExperimentResult.to_dict() carries an experiment id and a comparison block.
    assert "experiment_id" in payload
    assert "comparison" in payload


def test_main_framework_error_returns_exit_code_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A non-existent config directory triggers a ConfigError (AdaptiveHbError).
    code = cli.main(["build", "--config-dir", str(tmp_path / "missing")])
    assert code == 1
    assert "error:" in capsys.readouterr().err
