"""Evaluation module for computing and aggregating metrics."""

from .metrics import MetricsCollector, EpisodeMetrics, AggregateMetrics
from .evaluator import Evaluator, EvaluationResults
from .generalization import (
    GeneralizationEvaluator,
    GeneralizationResults,
    PerformanceDegradation,
)

__all__ = [
    "MetricsCollector",
    "EpisodeMetrics",
    "AggregateMetrics",
    "Evaluator",
    "EvaluationResults",
    "GeneralizationEvaluator",
    "GeneralizationResults",
    "PerformanceDegradation",
]
