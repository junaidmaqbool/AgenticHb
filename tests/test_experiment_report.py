"""Tests for the publication experiment-report generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptivehb.dataset import generate_synthetic_dataset
from adaptivehb.pipeline import HbPipeline
from adaptivehb.reporting.experiment_report import (
    ExperimentReportData,
    render_html,
    render_markdown,
    write_experiment_report,
)


def _rich_data() -> ExperimentReportData:
    return ExperimentReportData(
        name="real_run",
        experiment_id="real_run_20260707_abc123",
        metrics={
            "mae": 0.85, "rmse": 1.02, "r2": 0.78, "pearson": 0.88,
            "spearman": 0.86, "mean_bias": -0.05, "std_diff": 1.0,
            "clinical": {"within": {"0.5": 0.62, "1.0": 0.81}},
        },
        comparison={
            "metric": "mae", "baseline": 1.20, "adaptive": 0.85,
            "improvement": 0.35, "adaptive_better": True,
            "significance": {
                "n_pairs": 40, "mean_abs_error_diff": 0.35,
                "paired_t_test": {"p_value": 0.002, "t_statistic": 3.3, "df": 39},
                "wilcoxon": {"p_value": 0.004}, "cohens_d": 0.52,
                "bootstrap_ci": {"ci_lower": 0.15, "ci_upper": 0.55},
                "alpha": 0.05, "significant_at": True,
            },
        },
        provenance={
            "generated_at": "2026-07-07T10:00:00Z", "framework_version": "0.2.0", "seed": 42,
            "environment": {"python_version": "3.11.9", "platform": "Linux",
                            "packages": {"torch": "2.2.0", "numpy": "1.26.4"}},
            "git": {"short": "abc1234", "branch": "main", "detached": False},
            "config": {"digest": "dcd381ba8e88"},
            "dataset": {"available": True, "num_patients": 48, "num_samples": 336,
                        "split_sizes": {"train": 38, "test": 10}},
        },
        figures={"scatter": ["/x/figures/adaptive_scatter.png"],
                 "bland_altman": ["/x/figures/adaptive_bland_altman.png"]},
        num_test_patients=10,
    )


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #

def test_markdown_contains_core_sections() -> None:
    md = render_markdown(_rich_data())
    assert "# Experiment Report — real_run" in md
    assert "## Summary" in md
    assert "## Baseline vs adaptive" in md
    assert "## Adaptive metrics" in md
    assert "## Reproducibility" in md
    assert "## Figures" in md


def test_markdown_significance_statement_significant() -> None:
    md = render_markdown(_rich_data())
    assert "statistically significant" in md
    assert "not statistically significant" not in md
    assert "medium effect" in md  # d=0.52
    assert "Cohen's d" in md


def test_markdown_reports_provenance_and_metrics() -> None:
    md = render_markdown(_rich_data())
    assert "torch=2.2.0" in md
    assert "abc1234" in md and "main" in md
    assert "dcd381ba8e88" in md
    assert "Within ±0.5 g/dL" in md
    assert "62.0%" in md  # within-band fraction formatted as a percentage


def test_markdown_relative_figure_links(tmp_path: Path) -> None:
    data = _rich_data()
    data.figures = {"scatter": [str(tmp_path / "figures" / "adaptive_scatter.png")]}
    md = render_markdown(data, report_dir=tmp_path / "reports")
    assert "../figures/adaptive_scatter.png" in md


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

def test_html_is_wellformed_and_embeds_figures() -> None:
    html = render_html(_rich_data())
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert "<table>" in html
    assert "adaptive_scatter.png" in html
    assert "<img" in html


# --------------------------------------------------------------------------- #
# Degenerate / missing pieces
# --------------------------------------------------------------------------- #

def test_degenerate_significance_statement() -> None:
    data = ExperimentReportData(
        name="x", experiment_id="y",
        comparison={"metric": "mae", "baseline": 2.1, "adaptive": 2.1,
                    "improvement": 0.0, "adaptive_better": False,
                    "significance": {"n_pairs": 1, "paired_t_test": {"p_value": 1.0},
                                     "wilcoxon": {"p_value": 1.0}, "cohens_d": 0.0,
                                     "bootstrap_ci": {"ci_lower": 0.0, "ci_upper": 0.0}, "alpha": 0.05}},
    )
    md = render_markdown(data)
    assert "was not evaluable" in md


def test_empty_data_still_renders() -> None:
    md = render_markdown(ExperimentReportData(name="e", experiment_id="e"))
    assert "# Experiment Report — e" in md
    assert "No baseline-vs-adaptive comparison is available" in md
    assert "No figures were generated" in md
    # And HTML too.
    html = render_html(ExperimentReportData(name="e", experiment_id="e"))
    assert html.rstrip().endswith("</html>")


def test_write_experiment_report_writes_both(tmp_path: Path) -> None:
    paths = write_experiment_report(_rich_data(), tmp_path / "reports")
    assert Path(paths["markdown"]).is_file()
    assert Path(paths["html"]).is_file()
    assert Path(paths["markdown"]).read_text().startswith("# Experiment Report")


# --------------------------------------------------------------------------- #
# Experiment integration
# --------------------------------------------------------------------------- #

def test_experiment_archives_report(framework_config, tmp_path: Path) -> None:
    root = tmp_path / "ds"
    generate_synthetic_dataset(root, num_patients=10, seed=5)
    pipeline = HbPipeline(framework_config, base_dir=tmp_path, dataset_root=root)
    result = pipeline.experiment("reporttest", epochs=2)
    assert set(result.report_paths) == {"markdown", "html"}
    md = Path(result.report_paths["markdown"])
    html = Path(result.report_paths["html"])
    assert md.is_file() and html.is_file()
    assert md.parent.name == "reports"
    assert "Experiment Report" in md.read_text()
