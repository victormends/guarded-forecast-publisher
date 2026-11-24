"""Tests for Empirical Bayes partial-rollout shrinkage, Monte Carlo bounds, and convergence backtest."""

from __future__ import annotations

import math
from typing import List

import pytest

from guarded_forecast.shrinkage import (
    ShardObservation,
    estimate_empirical_bayes_rollout,
    run_convergence_backtest,
)


def test_empirical_bayes_shrinkage_weights_and_standard_error_formula() -> None:
    # 5 qualified shards + 1 below min_samples/fraction that must be excluded
    observations = [
        ShardObservation("us-east", f"s-{i}", reported_fraction=0.80, sample_size=500, observed_delta=0.10)
        for i in range(5)
    ] + [
        ShardObservation("us-east", "s-tiny", reported_fraction=0.10, sample_size=50, observed_delta=0.90)
    ]

    est = estimate_empirical_bayes_rollout(
        observations,
        fleet_prior_delta=0.0,
        prior_strength_k=10.0,
        min_reported_fraction=0.25,
        min_samples=300,
        se_floor=0.003,
        se_scale=0.012,
        monte_carlo_draws=400,
        seed=123,
    )

    assert est.qualified_shards == 5
    reg = est.regions["us-east"]
    assert reg.qualified_shards == 5
    # lambda = 5 / (5 + 10) = 1/3 -> shrunk = (1/3)*0.10 + (2/3)*0.0 = 0.033333
    assert reg.lambda_weight == pytest.approx(5.0 / 15.0, abs=1e-5)
    assert reg.shrunk_mean == pytest.approx(0.10 / 3.0, abs=1e-5)
    expected_se = max(0.003, 0.012 / math.sqrt(5.0))
    assert reg.standard_error == pytest.approx(expected_se, abs=1e-5)
    assert est.ci95_low < est.fleet_mean < est.ci95_high


def test_deterministic_monte_carlo_reproducibility_and_slo_breach_prob() -> None:
    obs = [
        ShardObservation("reg-1", f"s-{i}", reported_fraction=0.9, sample_size=400, observed_delta=0.05)
        for i in range(20)
    ]
    a = estimate_empirical_bayes_rollout(
        obs,
        fleet_prior_delta=0.03,
        slo_threshold_delta=0.025,
        monte_carlo_draws=300,
        seed=777,
    )
    b = estimate_empirical_bayes_rollout(
        obs,
        fleet_prior_delta=0.03,
        slo_threshold_delta=0.025,
        monte_carlo_draws=300,
        seed=777,
    )
    assert a == b
    assert a.slo_breach_prob == pytest.approx(1.0, abs=1e-3)


def test_skewed_partial_rollout_shrinkage_beats_naive_mean_early() -> None:
    events: List[ShardObservation] = []
    # Fast canary cohort arrives first (small share of capacity, larger early anomaly +0.045)
    for i in range(18):
        events.append(
            ShardObservation("fast-canary", f"fc-{i}", reported_fraction=1.0, sample_size=400, observed_delta=0.045)
        )
    # Larger baseline regions report later near baseline (+0.005)
    for reg in ("core-a", "core-b", "core-c"):
        for i in range(18):
            events.append(
                ShardObservation(reg, f"{reg}-{i}", reported_fraction=1.0, sample_size=500, observed_delta=0.005)
            )

    weights = {"fast-canary": 1.0, "core-a": 4.0, "core-b": 4.0, "core-c": 4.0}
    true_delta = (1.0 * 0.045 + 12.0 * 0.005) / 13.0

    rows = run_convergence_backtest(
        events,
        true_fleet_delta=true_delta,
        fleet_prior_delta=0.005,
        checkpoints=(0.05, 0.25, 0.50, 1.0),
        prior_strength_k=6.0,
        region_weights=weights,
        seed=99,
    )

    assert len(rows) == 4
    early_5pct, early_25pct, _, final_100pct = rows
    assert early_5pct.raw_abs_error is not None
    assert early_25pct.raw_abs_error is not None
    # Shrinkage anchors unseen core regions to prior instead of extrapolating fast canary outlier
    assert early_5pct.shrunk_abs_error < early_5pct.raw_abs_error
    assert early_25pct.shrunk_abs_error < early_25pct.raw_abs_error
    assert final_100pct.shrunk_abs_error < early_5pct.shrunk_abs_error
    assert final_100pct.true_delta_covered is True
