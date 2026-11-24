"""Guarded capacity quantile publisher and Empirical Bayes partial-rollout estimator."""

from __future__ import annotations

from .contract import (
    MODEL_KIND,
    REQUIRED_QUANTILES,
    ForecastModelError,
    canonical_json,
    input_fingerprint,
    load_validated_model,
)
from .publisher import (
    atomic_write_json,
    evaluate_quantiles,
    extract_features,
    issue_forecast,
    publish_if_inputs_changed,
)
from .shrinkage import (
    CheckpointComparison,
    RegionShrinkageEstimate,
    RolloutEstimate,
    ShardObservation,
    estimate_empirical_bayes_rollout,
    run_convergence_backtest,
)

__all__ = [
    "CheckpointComparison",
    "ForecastModelError",
    "MODEL_KIND",
    "REQUIRED_QUANTILES",
    "RegionShrinkageEstimate",
    "RolloutEstimate",
    "ShardObservation",
    "atomic_write_json",
    "canonical_json",
    "estimate_empirical_bayes_rollout",
    "evaluate_quantiles",
    "extract_features",
    "input_fingerprint",
    "issue_forecast",
    "load_validated_model",
    "publish_if_inputs_changed",
    "run_convergence_backtest",
]
