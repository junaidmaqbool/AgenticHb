"""Evaluation subsystem.

Provides torch-free metrics (regression, clinical agreement, classification,
calibration), paired significance testing for the baseline-vs-adaptive
comparison, a typed configuration, an :class:`EvaluationReport` with CSV/JSON
export, and the :class:`EvaluationManager` that ties them together.
"""

from adaptivehb.evaluation.config import EvaluationConfig
from adaptivehb.evaluation.manager import EvaluationManager
from adaptivehb.evaluation.report import EvaluationReport
from adaptivehb.evaluation.significance import compare_significance

__all__ = [
    "EvaluationManager",
    "EvaluationConfig",
    "EvaluationReport",
    "compare_significance",
]
