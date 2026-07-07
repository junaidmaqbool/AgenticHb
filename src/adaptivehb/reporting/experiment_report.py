"""Publication-ready experiment report generator (Decision 034).

An archived experiment already contains the raw evidence — metric JSONs, the
baseline-vs-adaptive comparison with paired significance, the provenance manifest,
per-sample predictions, and figures. This module consolidates that evidence into a
single, human-readable report a reader can drop into a paper's results section:

* a **Markdown** report (`experiment_report.md`) that renders on GitHub and in any
  Markdown viewer, and
* a **self-contained HTML** report (`experiment_report.html`) that embeds the
  figures inline via relative ``<img>`` links.

Both are produced from the same structured data by pure standard-library string
building — no matplotlib, pandas, or Markdown/HTML dependency — so report
generation never fails for want of an optional package and stays torch-free. The
report is robust to missing pieces: absent significance, provenance, dataset info,
or figures each degrade to a clear note rather than raising.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptivehb.core.utils import ensure_dir, write_json  # noqa: F401 (write_json re-exported for callers)

# Regression metrics shown, in order, with human labels and units.
_METRIC_LABELS: tuple[tuple[str, str], ...] = (
    ("mae", "MAE (g/dL)"),
    ("rmse", "RMSE (g/dL)"),
    ("r2", "R²"),
    ("pearson", "Pearson r"),
    ("spearman", "Spearman ρ"),
    ("mean_bias", "Mean bias (g/dL)"),
    ("std_diff", "SD of residuals (g/dL)"),
)


@dataclass
class ExperimentReportData:
    """Structured inputs for an experiment report.

    Every field except ``name``/``experiment_id`` is optional so a report can be
    produced from whatever an experiment archived.
    """

    name: str
    experiment_id: str
    metrics: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, list[str]] = field(default_factory=dict)
    num_test_patients: int | None = None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def render_markdown(data: ExperimentReportData, *, report_dir: Path | None = None) -> str:
    """Render the experiment report as Markdown.

    Args:
        data: The structured report inputs.
        report_dir: Directory the report will be written to; used to compute
            relative figure links. When ``None``, figure paths are used as-is.

    Returns:
        The Markdown document as a string.
    """
    lines: list[str] = []
    add = lines.append

    add(f"# Experiment Report — {data.name}")
    add("")
    add(f"- **Experiment id:** `{data.experiment_id}`")
    if data.provenance.get("generated_at"):
        add(f"- **Generated:** {data.provenance['generated_at']}")
    if data.provenance.get("framework_version"):
        add(f"- **Framework version:** {data.provenance['framework_version']}")
    if data.provenance.get("seed") is not None:
        add(f"- **Seed:** {data.provenance['seed']}")
    if data.num_test_patients is not None:
        add(f"- **Test patients:** {data.num_test_patients}")
    add("")

    # -- Summary / significance statement --------------------------------- #
    add("## Summary")
    add("")
    add(_comparison_statement(data.comparison))
    add("")

    # -- Comparison table ------------------------------------------------- #
    comparison = data.comparison
    if comparison:
        metric = str(comparison.get("metric", "mae")).upper()
        add("## Baseline vs adaptive")
        add("")
        add("| Pipeline | " + metric + " |")
        add("| --- | --- |")
        add(f"| Static baseline | {_fmt(comparison.get('baseline'))} |")
        add(f"| Adaptive (agent-fused) | {_fmt(comparison.get('adaptive'))} |")
        add(f"| Improvement | {_fmt(comparison.get('improvement'))} |")
        add("")
        significance = comparison.get("significance")
        if significance:
            add(_significance_table_markdown(significance))
            add("")

    # -- Metric bundle ---------------------------------------------------- #
    if data.metrics:
        add("## Adaptive metrics")
        add("")
        add("| Metric | Value |")
        add("| --- | --- |")
        for key, label in _METRIC_LABELS:
            if key in data.metrics:
                add(f"| {label} | {_fmt(data.metrics[key])} |")
        for band, value in _within_bands(data.metrics).items():
            add(f"| Within ±{band} g/dL | {_fmt_pct(value)} |")
        add("")

    # -- Provenance ------------------------------------------------------- #
    add("## Reproducibility")
    add("")
    add(_provenance_markdown(data.provenance))
    add("")

    # -- Figures ---------------------------------------------------------- #
    figure_paths = _figure_list(data.figures)
    add("## Figures")
    add("")
    if figure_paths:
        for title, path in figure_paths:
            rel = _relative(path, report_dir)
            add(f"### {title}")
            add("")
            add(f"![{title}]({rel})")
            add("")
    else:
        add("_No figures were generated (matplotlib not available at run time)._")
        add("")

    return "\n".join(lines).rstrip() + "\n"


def render_html(data: ExperimentReportData, *, report_dir: Path | None = None) -> str:
    """Render the experiment report as a self-contained HTML document."""
    esc = html.escape
    parts: list[str] = []
    add = parts.append

    add("<!DOCTYPE html>")
    add('<html lang="en"><head><meta charset="utf-8">')
    add(f"<title>Experiment Report — {esc(data.name)}</title>")
    add("<style>"
        "body{font-family:system-ui,Arial,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.5}"
        "table{border-collapse:collapse;margin:1rem 0}th,td{border:1px solid #ccc;padding:.35rem .6rem;text-align:left}"
        "th{background:#f4f4f4}code{background:#f4f4f4;padding:.1rem .3rem;border-radius:3px}"
        "img{max-width:100%;height:auto;border:1px solid #eee}figure{margin:1rem 0}"
        "</style></head><body>")

    add(f"<h1>Experiment Report — {esc(data.name)}</h1>")
    add("<ul>")
    add(f"<li><strong>Experiment id:</strong> <code>{esc(data.experiment_id)}</code></li>")
    if data.provenance.get("generated_at"):
        add(f"<li><strong>Generated:</strong> {esc(str(data.provenance['generated_at']))}</li>")
    if data.provenance.get("framework_version"):
        add(f"<li><strong>Framework version:</strong> {esc(str(data.provenance['framework_version']))}</li>")
    if data.provenance.get("seed") is not None:
        add(f"<li><strong>Seed:</strong> {esc(str(data.provenance['seed']))}</li>")
    if data.num_test_patients is not None:
        add(f"<li><strong>Test patients:</strong> {esc(str(data.num_test_patients))}</li>")
    add("</ul>")

    add("<h2>Summary</h2>")
    add(f"<p>{esc(_comparison_statement(data.comparison))}</p>")

    comparison = data.comparison
    if comparison:
        metric = esc(str(comparison.get("metric", "mae")).upper())
        add("<h2>Baseline vs adaptive</h2>")
        add("<table><tr><th>Pipeline</th><th>" + metric + "</th></tr>")
        add(f"<tr><td>Static baseline</td><td>{_fmt(comparison.get('baseline'))}</td></tr>")
        add(f"<tr><td>Adaptive (agent-fused)</td><td>{_fmt(comparison.get('adaptive'))}</td></tr>")
        add(f"<tr><td>Improvement</td><td>{_fmt(comparison.get('improvement'))}</td></tr>")
        add("</table>")
        significance = comparison.get("significance")
        if significance:
            add(_significance_table_html(significance))

    if data.metrics:
        add("<h2>Adaptive metrics</h2>")
        add("<table><tr><th>Metric</th><th>Value</th></tr>")
        for key, label in _METRIC_LABELS:
            if key in data.metrics:
                add(f"<tr><td>{esc(label)}</td><td>{_fmt(data.metrics[key])}</td></tr>")
        for band, value in _within_bands(data.metrics).items():
            add(f"<tr><td>Within ±{esc(str(band))} g/dL</td><td>{_fmt_pct(value)}</td></tr>")
        add("</table>")

    add("<h2>Reproducibility</h2>")
    add("<pre>" + esc(_provenance_markdown(data.provenance)) + "</pre>")

    add("<h2>Figures</h2>")
    figure_paths = _figure_list(data.figures)
    if figure_paths:
        for title, path in figure_paths:
            rel = esc(_relative(path, report_dir))
            add(f"<figure><figcaption><strong>{esc(title)}</strong></figcaption>"
                f'<img src="{rel}" alt="{esc(title)}"></figure>')
    else:
        add("<p><em>No figures were generated (matplotlib not available at run time).</em></p>")

    add("</body></html>")
    return "\n".join(parts) + "\n"


def write_experiment_report(data: ExperimentReportData, report_dir: str | Path) -> dict[str, str]:
    """Render and write both report formats into ``report_dir``.

    Returns:
        Mapping with the written ``markdown`` and ``html`` paths.
    """
    out_dir = ensure_dir(report_dir)
    md_path = out_dir / "experiment_report.md"
    html_path = out_dir / "experiment_report.html"
    md_path.write_text(render_markdown(data, report_dir=out_dir), encoding="utf-8")
    html_path.write_text(render_html(data, report_dir=out_dir), encoding="utf-8")
    return {"markdown": str(md_path), "html": str(html_path)}


# --------------------------------------------------------------------------- #
# Narrative helpers
# --------------------------------------------------------------------------- #

def _comparison_statement(comparison: dict[str, Any]) -> str:
    """A plain-language sentence describing the baseline-vs-adaptive result."""
    if not comparison:
        return "No baseline-vs-adaptive comparison is available for this experiment."
    metric = str(comparison.get("metric", "mae")).upper()
    baseline = comparison.get("baseline")
    adaptive = comparison.get("adaptive")
    improvement = comparison.get("improvement")
    better = comparison.get("adaptive_better")

    direction = "reduced" if better else ("increased" if (improvement or 0) < 0 else "did not change")
    pct = ""
    if isinstance(baseline, (int, float)) and baseline:
        pct = f" ({abs(improvement) / abs(baseline) * 100:.1f}%)" if isinstance(improvement, (int, float)) else ""
    sentence = (
        f"The adaptive pipeline {direction} {metric} from {_fmt(baseline)} (static baseline) to "
        f"{_fmt(adaptive)}, a change of {_fmt(improvement)} g/dL{pct}."
    )

    significance = comparison.get("significance")
    if significance:
        sentence += " " + _significance_statement(significance)
    return sentence


def _significance_statement(significance: dict[str, Any]) -> str:
    """A plain-language significance sentence from the significance bundle."""
    n = significance.get("n_pairs", 0)
    if not n or n < 2:
        return (
            "The paired significance test was not evaluable (fewer than two paired test "
            "patients); collect a larger test split for an inferential result."
        )
    t_test = significance.get("paired_t_test", {})
    p = t_test.get("p_value")
    wil = significance.get("wilcoxon", {}).get("p_value")
    d = significance.get("cohens_d")
    ci = significance.get("bootstrap_ci", {})
    alpha = significance.get("alpha", 0.05)
    verdict = "statistically significant" if (isinstance(p, (int, float)) and p < alpha) else "not statistically significant"
    return (
        f"Across {n} paired test patients this difference was {verdict} "
        f"(paired t-test p={_fmt(p, 4)}, Wilcoxon p={_fmt(wil, 4)}; Cohen's d={_fmt(d, 3)}, "
        f"{_effect_size_label(d)}; bootstrap {int(round((1 - alpha) * 100))}% CI "
        f"[{_fmt(ci.get('ci_lower'))}, {_fmt(ci.get('ci_upper'))}])."
    )


def _effect_size_label(d: Any) -> str:
    """Cohen's convention for interpreting |d|."""
    if not isinstance(d, (int, float)):
        return "effect size n/a"
    ad = abs(d)
    if ad < 0.2:
        return "negligible effect"
    if ad < 0.5:
        return "small effect"
    if ad < 0.8:
        return "medium effect"
    return "large effect"


def _significance_table_markdown(significance: dict[str, Any]) -> str:
    t = significance.get("paired_t_test", {})
    w = significance.get("wilcoxon", {})
    ci = significance.get("bootstrap_ci", {})
    rows = [
        "| Statistic | Value |",
        "| --- | --- |",
        f"| Paired samples (n) | {significance.get('n_pairs', 0)} |",
        f"| Mean |Δerror| (g/dL) | {_fmt(significance.get('mean_abs_error_diff'))} |",
        f"| Paired t-test p | {_fmt(t.get('p_value'), 4)} |",
        f"| Wilcoxon p | {_fmt(w.get('p_value'), 4)} |",
        f"| Cohen's d | {_fmt(significance.get('cohens_d'), 3)} ({_effect_size_label(significance.get('cohens_d'))}) |",
        f"| Bootstrap CI | [{_fmt(ci.get('ci_lower'))}, {_fmt(ci.get('ci_upper'))}] |",
    ]
    return "\n".join(rows)


def _significance_table_html(significance: dict[str, Any]) -> str:
    t = significance.get("paired_t_test", {})
    w = significance.get("wilcoxon", {})
    ci = significance.get("bootstrap_ci", {})
    return (
        "<table><tr><th>Statistic</th><th>Value</th></tr>"
        f"<tr><td>Paired samples (n)</td><td>{significance.get('n_pairs', 0)}</td></tr>"
        f"<tr><td>Mean |&Delta;error| (g/dL)</td><td>{_fmt(significance.get('mean_abs_error_diff'))}</td></tr>"
        f"<tr><td>Paired t-test p</td><td>{_fmt(t.get('p_value'), 4)}</td></tr>"
        f"<tr><td>Wilcoxon p</td><td>{_fmt(w.get('p_value'), 4)}</td></tr>"
        f"<tr><td>Cohen's d</td><td>{_fmt(significance.get('cohens_d'), 3)} "
        f"({_effect_size_label(significance.get('cohens_d'))})</td></tr>"
        f"<tr><td>Bootstrap CI</td><td>[{_fmt(ci.get('ci_lower'))}, {_fmt(ci.get('ci_upper'))}]</td></tr>"
        "</table>"
    )


def _provenance_markdown(provenance: dict[str, Any]) -> str:
    if not provenance:
        return "_No provenance manifest was recorded for this experiment._"
    lines: list[str] = []
    env = provenance.get("environment", {})
    git = provenance.get("git")
    config = provenance.get("config", {})
    dataset = provenance.get("dataset", {})
    if env:
        lines.append(f"- **Python:** {env.get('python_version', 'unknown')} on {env.get('platform', 'unknown')}")
        packages = env.get("packages", {})
        if packages:
            joined = ", ".join(f"{k}={v}" for k, v in sorted(packages.items()))
            lines.append(f"- **Packages:** {joined}")
    if git:
        commit = git.get("short") or git.get("commit") or "unknown"
        branch = git.get("branch") or ("detached" if git.get("detached") else "unknown")
        lines.append(f"- **Git:** {commit} ({branch})")
    else:
        lines.append("- **Git:** not a git checkout at run time")
    if config.get("digest"):
        lines.append(f"- **Config fingerprint:** `{config['digest']}`")
    if dataset and dataset.get("available"):
        lines.append(
            f"- **Dataset:** {dataset.get('num_patients')} patients, {dataset.get('num_samples')} samples, "
            f"splits {dataset.get('split_sizes')}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Small formatting utilities
# --------------------------------------------------------------------------- #

def _fmt(value: Any, digits: int = 4) -> str:
    """Format a numeric value to ``digits`` decimals; pass through non-numbers."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "n/a" if value is None else str(value)


def _fmt_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return "n/a"


def _within_bands(metrics: dict[str, Any]) -> dict[str, float]:
    """Extract the clinical within-band fractions, if present."""
    clinical = metrics.get("clinical") if isinstance(metrics, dict) else None
    if isinstance(clinical, dict):
        within = clinical.get("within")
        if isinstance(within, dict):
            return {str(k): float(v) for k, v in within.items()}
    return {}


def _figure_list(figures: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Flatten the figures mapping into ``(title, path)`` pairs."""
    pairs: list[tuple[str, str]] = []
    for name, paths in figures.items():
        title = name.replace("_", " ").title()
        for path in paths:
            if str(path).lower().endswith(".png"):
                pairs.append((title, str(path)))
    return pairs


def _relative(path: str, report_dir: Path | None) -> str:
    """Return ``path`` relative to ``report_dir`` when possible, else unchanged."""
    if report_dir is None:
        return path
    try:
        import os

        return os.path.relpath(path, report_dir)
    except (ValueError, OSError):
        return path


__all__ = [
    "ExperimentReportData",
    "render_markdown",
    "render_html",
    "write_experiment_report",
]
