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


def test_normal_normal_conjugate_posterior_exact_values() -> None:
    obs = [
        ShardObservation("reg-a", f"s-{i}", reported_fraction=1.0, sample_size=500, observed_delta=0.030)
        for i in range(4)
    ]
    prior_mu = 0.005
    prior_sigma = 0.012
    within_sigma = 0.018

    est = estimate_empirical_bayes_rollout(
        obs,
        fleet_prior_delta=prior_mu,
        within_shard_sigma=within_sigma,
        prior_sigma=prior_sigma,
        region_weights={"reg-a": 1.0, "reg-b": 3.0},
        monte_carlo_draws=500,
        seed=11,
    )

    var_prior = prior_sigma ** 2
    var_obs = (within_sigma ** 2) / 4.0
    expected_lam = var_prior / (var_prior + var_obs)
    expected_mean = expected_lam * 0.030 + (1.0 - expected_lam) * prior_mu
    expected_se = math.sqrt((1.0 - expected_lam) * var_prior)

    reg_a = est.regions["reg-a"]
    assert reg_a.lambda_weight == pytest.approx(expected_lam, abs=1e-6)
    assert reg_a.shrunk_mean == pytest.approx(expected_mean, abs=1e-6)
    assert reg_a.standard_error == pytest.approx(expected_se, abs=1e-6)

    reg_b = est.regions["reg-b"]
    assert reg_b.qualified_shards == 0
    assert reg_b.lambda_weight == 0.0
    assert reg_b.shrunk_mean == pytest.approx(prior_mu, abs=1e-6)
    assert reg_b.standard_error == pytest.approx(prior_sigma, abs=1e-6)


def test_conjugate_eb_finite_population_convergence_at_100pct() -> None:
    obs = [
        ShardObservation("reg-a", f"a-{i}", reported_fraction=1.0, sample_size=500, observed_delta=0.025)
        for i in range(10)
    ] + [
        ShardObservation("reg-b", f"b-{i}", reported_fraction=1.0, sample_size=500, observed_delta=-0.010)
        for i in range(10)
    ]
    est = estimate_empirical_bayes_rollout(
        obs,
        fleet_prior_delta=0.005,
        within_shard_sigma=0.018,
        prior_sigma=0.012,
        region_total_shards={"reg-a": 10, "reg-b": 10},
        monte_carlo_draws=300,
        seed=22,
    )

    for reg_id, expected_raw in (("reg-a", 0.025), ("reg-b", -0.010)):
        reg = est.regions[reg_id]
        assert reg.lambda_weight == 1.0
        assert reg.shrunk_mean == pytest.approx(expected_raw, abs=1e-6)
        assert reg.standard_error == 0.0


def test_hierarchical_200_seed_monte_carlo_ci_calibration() -> None:
    import numpy as np

    weights = {
        "region-canary-fast": 1.0,
        "region-core-a": 4.0,
        "region-core-b": 4.0,
        "region-core-c": 4.0,
    }
    w_tot = sum(weights.values())
    mu0 = 0.005
    tau0 = 0.012
    sigma_obs = 0.018
    checkpoints = (0.05, 0.25, 0.50, 1.0)
    covered = {cp: 0 for cp in checkpoints}

    for seed in range(1, 201):
        rng = np.random.default_rng(seed)
        thetas = {r: float(rng.normal(mu0, tau0)) for r in weights}
        true_delta = sum(weights[r] * thetas[r] for r in weights) / w_tot
        events = [
            ShardObservation(r, f"{r}-{i}", 1.0, 500, round(float(rng.normal(thetas[r], sigma_obs)), 6))
            for r in ("region-canary-fast", "region-core-a", "region-core-b", "region-core-c")
            for i in range(15)
        ]
        rows = run_convergence_backtest(
            events,
            true_fleet_delta=true_delta,
            fleet_prior_delta=mu0,
            checkpoints=checkpoints,
            within_shard_sigma=sigma_obs,
            prior_sigma=tau0,
            region_weights=weights,
            monte_carlo_draws=1000,
            seed=seed,
        )
        for row in rows:
            if row.true_delta_covered:
                covered[row.checkpoint_fraction] += 1

    for cp in checkpoints:
        rate = covered[cp] / 200.0
        assert 0.90 <= rate <= 0.98, f"Unreliable 95% CI coverage at {cp}: {rate}"
