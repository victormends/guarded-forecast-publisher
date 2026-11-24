"""Hierarchical Empirical Bayes shrinkage and Monte Carlo intervals for progressive canary rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ShardObservation:
    """Telemetry shift measured on one rollout shard/canary cohort."""

    region_id: str
    shard_id: str
    reported_fraction: float
    sample_size: int
    observed_delta: float
    weight: float = 1.0


@dataclass(frozen=True)
class RegionShrinkageEstimate:
    """Regional posterior estimate combining local shards with the global fleet prior."""

    region_id: str
    qualified_shards: int
    lambda_weight: float
    raw_mean: Optional[float]
    shrunk_mean: float
    standard_error: float


@dataclass(frozen=True)
class RolloutEstimate:
    """Aggregated fleet delta estimate with Monte Carlo credible intervals."""

    fleet_mean: float
    raw_fleet_mean: Optional[float]
    ci95_low: float
    ci95_high: float
    slo_breach_prob: float
    qualified_shards: int
    total_observations: int
    regions: Dict[str, RegionShrinkageEstimate] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointComparison:
    """Walk-forward convergence accuracy at one rollout reporting fraction."""

    checkpoint_fraction: float
    reported_shards: int
    raw_mean: Optional[float]
    raw_abs_error: Optional[float]
    shrunk_mean: float
    shrunk_abs_error: float
    ci95_low: float
    ci95_high: float
    true_delta_covered: bool


def estimate_empirical_bayes_rollout(
    observations: Sequence[ShardObservation],
    *,
    fleet_prior_delta: float = 0.0,
    prior_strength_k: float = 10.0,
    min_reported_fraction: float = 0.25,
    min_samples: int = 300,
    se_floor: float = 0.003,
    se_scale: float = 0.012,
    fallback_prior_se: float = 0.015,
    slo_threshold_delta: float = 0.020,
    monte_carlo_draws: int = 200,
    seed: int = 42,
    region_weights: Optional[Mapping[str, float]] = None,
    within_shard_sigma: Optional[float] = None,
    prior_sigma: Optional[float] = None,
    region_total_shards: Optional[Mapping[str, int]] = None,
) -> RolloutEstimate:
    """Estimate fleet metric shift from partial/skewed canary rollout reports.

    Supports two shrinkage modes:
    1. Exact Normal-Normal conjugate Empirical Bayes (when `within_shard_sigma`
       and `prior_sigma` are provided): computes regional weight
       `lambda = tau0^2 / (tau0^2 + (sigma_obs^2 / n) * fpc)` and posterior
       standard error `sqrt((1 - lambda) * tau0^2)`, with optional finite-population
       correction via `region_total_shards`.
    2. Legacy heuristic mode (when either sigma argument is None): shrinks via
       `lambda = n / (n + k)` and `SE = max(se_floor, se_scale / sqrt(n))`.
    """
    if prior_strength_k <= 0:
        raise ValueError("prior_strength_k must be > 0")
    if monte_carlo_draws < 10:
        raise ValueError("monte_carlo_draws must be >= 10")
    if not (0.0 <= min_reported_fraction <= 1.0):
        raise ValueError("min_reported_fraction must be in [0, 1]")
    if within_shard_sigma is not None and within_shard_sigma <= 0:
        raise ValueError("within_shard_sigma must be > 0")
    if prior_sigma is not None and prior_sigma <= 0:
        raise ValueError("prior_sigma must be > 0")
    if region_total_shards is not None:
        for reg_name, reg_total in region_total_shards.items():
            if reg_total <= 0:
                raise ValueError(f"region_total_shards[{reg_name!r}] must be > 0, got {reg_total}")

    use_conjugate_eb = within_shard_sigma is not None and prior_sigma is not None
    effective_fallback_se = float(prior_sigma) if use_conjugate_eb else float(fallback_prior_se)

    by_region: Dict[str, List[ShardObservation]] = {}
    all_regions: set = set()
    for obs in observations:
        all_regions.add(obs.region_id)
        if obs.reported_fraction >= min_reported_fraction and obs.sample_size >= min_samples:
            by_region.setdefault(obs.region_id, []).append(obs)

    if region_weights:
        all_regions.update(region_weights.keys())

    if not all_regions:
        return RolloutEstimate(
            fleet_mean=round(float(fleet_prior_delta), 6),
            raw_fleet_mean=None,
            ci95_low=round(float(fleet_prior_delta - 1.96 * effective_fallback_se), 6),
            ci95_high=round(float(fleet_prior_delta + 1.96 * effective_fallback_se), 6),
            slo_breach_prob=0.0,
            qualified_shards=0,
            total_observations=len(observations),
            regions={},
        )

    qualified_all: List[ShardObservation] = [
        obs for items in by_region.values() for obs in items
    ]
    raw_fleet_mean: Optional[float] = None
    if qualified_all:
        w_sum = sum(max(1e-9, float(o.weight)) for o in qualified_all)
        raw_fleet_mean = round(
            sum(max(1e-9, float(o.weight)) * float(o.observed_delta) for o in qualified_all)
            / w_sum,
            6,
        )

    region_estimates: Dict[str, RegionShrinkageEstimate] = {}
    for region_id in sorted(all_regions):
        q_obs = by_region.get(region_id, [])
        n_reg = len(q_obs)
        if use_conjugate_eb:
            assert within_shard_sigma is not None and prior_sigma is not None
            if n_reg == 0:
                raw_mean = None
                lam = 0.0
                shrunk_mean = float(fleet_prior_delta)
                se = float(prior_sigma)
            else:
                raw_mean = float(sum(float(o.observed_delta) for o in q_obs) / n_reg)
                n_tot = region_total_shards.get(region_id) if region_total_shards else None
                fpc = (
                    max(0.0, 1.0 - n_reg / float(n_tot))
                    if n_tot is not None and n_tot > 0
                    else 1.0
                )
                var_obs = ((float(within_shard_sigma) ** 2) / float(n_reg)) * fpc
                var_prior = float(prior_sigma) ** 2
                if var_obs <= 0.0:
                    lam = 1.0
                    shrunk_mean = raw_mean
                    se = 0.0
                else:
                    lam = var_prior / (var_prior + var_obs)
                    shrunk_mean = lam * raw_mean + (1.0 - lam) * float(fleet_prior_delta)
                    se = math.sqrt((1.0 - lam) * var_prior)
        else:
            if n_reg > 0:
                raw_mean = float(sum(float(o.observed_delta) for o in q_obs) / n_reg)
                lam = float(n_reg) / (float(n_reg) + float(prior_strength_k))
                shrunk_mean = lam * raw_mean + (1.0 - lam) * float(fleet_prior_delta)
                se = max(float(se_floor), float(se_scale) / math.sqrt(float(n_reg)))
            else:
                raw_mean = None
                lam = 0.0
                shrunk_mean = float(fleet_prior_delta)
                se = float(fallback_prior_se)

        region_estimates[region_id] = RegionShrinkageEstimate(
            region_id=region_id,
            qualified_shards=n_reg,
            lambda_weight=round(lam, 6),
            raw_mean=round(raw_mean, 6) if raw_mean is not None else None,
            shrunk_mean=round(shrunk_mean, 6),
            standard_error=round(se, 6),
        )

    # Compute weighted fleet point estimate + deterministic Monte Carlo intervals.
    ordered_regions = sorted(region_estimates.keys())
    weights = np.array(
        [
            float(region_weights.get(r, 1.0)) if region_weights else 1.0
            for r in ordered_regions
        ],
        dtype=float,
    )
    weight_total = float(np.sum(weights))
    if weight_total <= 0:
        raise ValueError("sum of region_weights must be positive")
    weights = weights / weight_total

    means = np.array([region_estimates[r].shrunk_mean for r in ordered_regions], dtype=float)
    sigmas = np.array([region_estimates[r].standard_error for r in ordered_regions], dtype=float)

    rng = np.random.default_rng(seed)
    draws = rng.normal(loc=means, scale=sigmas, size=(monte_carlo_draws, len(ordered_regions)))
    fleet_draws = draws @ weights

    fleet_point = float(means @ weights)
    ci_low = float(np.percentile(fleet_draws, 2.5))
    ci_high = float(np.percentile(fleet_draws, 97.5))
    breach_prob = float(np.mean(fleet_draws >= float(slo_threshold_delta)))

    return RolloutEstimate(
        fleet_mean=round(fleet_point, 6),
        raw_fleet_mean=raw_fleet_mean,
        ci95_low=round(ci_low, 6),
        ci95_high=round(ci_high, 6),
        slo_breach_prob=round(breach_prob, 4),
        qualified_shards=len(qualified_all),
        total_observations=len(observations),
        regions=region_estimates,
    )


def run_convergence_backtest(
    shard_events: Sequence[ShardObservation],
    *,
    true_fleet_delta: float,
    fleet_prior_delta: float = 0.0,
    checkpoints: Sequence[float] = (0.05, 0.25, 0.50, 1.0),
    prior_strength_k: float = 10.0,
    min_reported_fraction: float = 0.25,
    min_samples: int = 300,
    monte_carlo_draws: int = 250,
    seed: int = 42,
    region_weights: Optional[Mapping[str, float]] = None,
    within_shard_sigma: Optional[float] = None,
    prior_sigma: Optional[float] = None,
    region_total_shards: Optional[Mapping[str, int]] = None,
) -> List[CheckpointComparison]:
    """Replay skewed shard arrival order across rollout checkpoints and score convergence."""
    if not shard_events:
        raise ValueError("shard_events must be non-empty")

    total = len(shard_events)
    all_region_weights: Dict[str, float] = (
        dict(region_weights)
        if region_weights
        else {r: 1.0 for r in sorted({o.region_id for o in shard_events})}
    )

    results: List[CheckpointComparison] = []
    for cp in checkpoints:
        if not (0.0 < cp <= 1.0):
            raise ValueError(f"checkpoint fraction must be in (0, 1], got {cp}")
        cutoff = max(1, min(total, int(math.ceil(total * float(cp)))))
        subset = shard_events[:cutoff]
        est = estimate_empirical_bayes_rollout(
            subset,
            fleet_prior_delta=fleet_prior_delta,
            prior_strength_k=prior_strength_k,
            min_reported_fraction=min_reported_fraction,
            min_samples=min_samples,
            monte_carlo_draws=monte_carlo_draws,
            seed=seed,
            region_weights=all_region_weights,
            within_shard_sigma=within_shard_sigma,
            prior_sigma=prior_sigma,
            region_total_shards=region_total_shards,
        )
        raw_err = (
            round(abs(est.raw_fleet_mean - true_fleet_delta), 6)
            if est.raw_fleet_mean is not None
            else None
        )
        shrunk_err = round(abs(est.fleet_mean - true_fleet_delta), 6)
        covered = est.ci95_low <= true_fleet_delta <= est.ci95_high
        results.append(
            CheckpointComparison(
                checkpoint_fraction=float(cp),
                reported_shards=cutoff,
                raw_mean=est.raw_fleet_mean,
                raw_abs_error=raw_err,
                shrunk_mean=est.fleet_mean,
                shrunk_abs_error=shrunk_err,
                ci95_low=est.ci95_low,
                ci95_high=est.ci95_high,
                true_delta_covered=covered,
            )
        )
    return results
