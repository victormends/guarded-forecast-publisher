"""Runnable local demo: contract-gated quantile publishing and 200-seed Monte Carlo rollout replay."""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from guarded_forecast import (
    MODEL_KIND,
    ForecastModelError,
    ShardObservation,
    estimate_empirical_bayes_rollout,
    publish_if_inputs_changed,
    run_convergence_backtest,
)


def _approved_model_artifact() -> Dict[str, Any]:
    return {
        "kind": MODEL_KIND,
        "model_id": "capacity-quantile-linear",
        "model_version": "2026.11.1",
        "review_status": "approved",
        "validation": {
            "sample_count": 240,
            "p10_p90_coverage": 0.84,
            "eval_window": "2026-Q3",
        },
        "feature_defaults": {
            "cpu_util_pct": 45.0,
            "queue_depth": 12.0,
        },
        "horizons": [
            {
                "hours": 1,
                "coefficients": {
                    "p10": {"intercept": 5.0, "weights": {"cpu_util_pct": 0.70, "queue_depth": 0.15}},
                    "p25": {"intercept": 7.0, "weights": {"cpu_util_pct": 0.78, "queue_depth": 0.20}},
                    "p50": {"intercept": 10.0, "weights": {"cpu_util_pct": 0.85, "queue_depth": 0.25}},
                    "p75": {"intercept": 13.0, "weights": {"cpu_util_pct": 0.92, "queue_depth": 0.30}},
                    "p90": {"intercept": 16.0, "weights": {"cpu_util_pct": 0.98, "queue_depth": 0.35}},
                },
            },
            {
                "hours": 6,
                "coefficients": {
                    "p10": {"intercept": 6.0, "weights": {"cpu_util_pct": 0.72, "queue_depth": 0.18}},
                    "p25": {"intercept": 8.5, "weights": {"cpu_util_pct": 0.80, "queue_depth": 0.24}},
                    "p50": {"intercept": 12.0, "weights": {"cpu_util_pct": 0.88, "queue_depth": 0.30}},
                    "p75": {"intercept": 15.5, "weights": {"cpu_util_pct": 0.95, "queue_depth": 0.36}},
                    "p90": {"intercept": 19.0, "weights": {"cpu_util_pct": 1.02, "queue_depth": 0.42}},
                },
            },
        ],
    }


def _generate_hierarchical_seed_events(
    seed: int,
    *,
    fleet_prior_delta: float,
    prior_sigma: float,
    within_shard_sigma: float,
    region_weights: Dict[str, float],
    shards_per_region: int = 15,
) -> Tuple[List[ShardObservation], float]:
    rng = np.random.default_rng(seed)
    region_order = (
        "region-canary-fast",
        "region-core-a",
        "region-core-b",
        "region-core-c",
    )
    regional_thetas = {
        reg: float(rng.normal(loc=fleet_prior_delta, scale=prior_sigma))
        for reg in region_order
    }
    total_w = sum(region_weights[reg] for reg in region_order)
    true_fleet_delta = sum(region_weights[reg] * regional_thetas[reg] for reg in region_order) / total_w

    # Skewed rollout arrival order: fast canary cohort reports all shards first.
    events: List[ShardObservation] = []
    for reg in region_order:
        theta_r = regional_thetas[reg]
        for i in range(shards_per_region):
            delta = float(rng.normal(loc=theta_r, scale=within_shard_sigma))
            events.append(
                ShardObservation(
                    region_id=reg,
                    shard_id=f"{reg}-{i}",
                    reported_fraction=1.0,
                    sample_size=500,
                    observed_delta=round(delta, 6),
                )
            )
    return events, true_fleet_delta


def run_monte_carlo_evaluation(
    seeds: range = range(1, 201),
    checkpoints: tuple = (0.05, 0.25, 0.50, 1.0),
) -> Dict[str, Any]:
    weights = {
        "region-canary-fast": 1.0,
        "region-core-a": 4.0,
        "region-core-b": 4.0,
        "region-core-c": 4.0,
    }
    fleet_prior_delta = 0.005
    prior_sigma = 0.012
    within_shard_sigma = 0.018
    shards_per_region = 15

    per_cp_raw_err: Dict[float, List[float]] = {cp: [] for cp in checkpoints}
    per_cp_eb_err: Dict[float, List[float]] = {cp: [] for cp in checkpoints}
    per_cp_covered: Dict[float, List[bool]] = {cp: [] for cp in checkpoints}
    per_cp_shards: Dict[float, int] = {}

    last_events: List[ShardObservation] = []
    for seed in seeds:
        events, true_delta = _generate_hierarchical_seed_events(
            seed,
            fleet_prior_delta=fleet_prior_delta,
            prior_sigma=prior_sigma,
            within_shard_sigma=within_shard_sigma,
            region_weights=weights,
            shards_per_region=shards_per_region,
        )
        last_events = events
        rows = run_convergence_backtest(
            events,
            true_fleet_delta=true_delta,
            fleet_prior_delta=fleet_prior_delta,
            checkpoints=checkpoints,
            within_shard_sigma=within_shard_sigma,
            prior_sigma=prior_sigma,
            region_weights=weights,
            monte_carlo_draws=1000,
            seed=seed,
        )
        for row in rows:
            cp = row.checkpoint_fraction
            per_cp_shards[cp] = row.reported_shards
            if row.raw_abs_error is not None:
                per_cp_raw_err[cp].append(row.raw_abs_error)
            per_cp_eb_err[cp].append(row.shrunk_abs_error)
            per_cp_covered[cp].append(row.true_delta_covered)

    # Verify finite-population convergence when region_total_shards reaches 100% reporting.
    fpc_totals = {reg: shards_per_region for reg in weights}
    fpc_est = estimate_empirical_bayes_rollout(
        last_events,
        fleet_prior_delta=fleet_prior_delta,
        within_shard_sigma=within_shard_sigma,
        prior_sigma=prior_sigma,
        region_weights=weights,
        region_total_shards=fpc_totals,
        monte_carlo_draws=1000,
        seed=200,
    )
    fpc_check = {
        reg: {
            "lambda_weight": reg_est.lambda_weight,
            "raw_mean": reg_est.raw_mean,
            "shrunk_mean": reg_est.shrunk_mean,
            "standard_error": reg_est.standard_error,
        }
        for reg, reg_est in fpc_est.regions.items()
    }

    summary_rows: List[Dict[str, Any]] = []
    seed_count = len(list(seeds))
    for cp in checkpoints:
        raw_arr = np.array(per_cp_raw_err[cp], dtype=float)
        eb_arr = np.array(per_cp_eb_err[cp], dtype=float)
        cov_arr = np.array(per_cp_covered[cp], dtype=bool)
        summary_rows.append(
            {
                "checkpoint_pct": int(round(cp * 100)),
                "reported_shards": per_cp_shards[cp],
                "raw_mae": round(float(np.mean(raw_arr)), 6),
                "eb_mae": round(float(np.mean(eb_arr)), 6),
                "raw_median_ae": round(float(np.median(raw_arr)), 6),
                "eb_median_ae": round(float(np.median(eb_arr)), 6),
                "ci95_coverage_rate": round(float(np.mean(cov_arr)), 4),
                "ci95_covered_seeds": f"{int(np.sum(cov_arr))}/{seed_count}",
            }
        )

    return {
        "seeds": f"{min(seeds)}..{max(seeds)} ({seed_count} seeds)",
        "model": {
            "fleet_prior_delta": fleet_prior_delta,
            "prior_sigma": prior_sigma,
            "within_shard_sigma": within_shard_sigma,
            "total_shards": shards_per_region * len(weights),
        },
        "checkpoints": summary_rows,
        "finite_population_100pct_check": fpc_check,
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="guarded-forecast-demo-") as tmp:
        root = Path(tmp)

        # 1. Fail-closed contract check: draft / unapproved model must be rejected.
        draft_path = root / "model_draft.json"
        draft = _approved_model_artifact()
        draft["review_status"] = "draft"
        draft_path.write_text(json.dumps(draft), encoding="utf-8")

        rejected_draft = False
        try:
            publish_if_inputs_changed(
                model_path=draft_path,
                input_snapshot={"cluster_id": "prod-east", "metrics": {"cpu_util_pct": 62.0}},
                output_path=root / "out" / "forecast.json",
                state_path=root / "state" / "publish_state.json",
            )
        except ForecastModelError:
            rejected_draft = True
        assert rejected_draft, "Draft model artifact must never publish"

        # 2. Approved model publishes atomically, then skips unchanged SHA-256 fingerprint.
        approved_path = root / "model_approved.json"
        approved_path.write_text(json.dumps(_approved_model_artifact()), encoding="utf-8")

        snapshot = {"cluster_id": "prod-east", "metrics": {"cpu_util_pct": 68.5, "queue_depth": 18.0}}
        issued_at = datetime(2026, 11, 24, 12, 0, 0, tzinfo=timezone.utc)
        first = publish_if_inputs_changed(
            model_path=approved_path,
            input_snapshot=snapshot,
            output_path=root / "out" / "forecast.json",
            state_path=root / "state" / "publish_state.json",
            issued_at=issued_at,
        )
        second = publish_if_inputs_changed(
            model_path=approved_path,
            input_snapshot=snapshot,
            output_path=root / "out" / "forecast.json",
            state_path=root / "state" / "publish_state.json",
            issued_at=issued_at,
        )
        assert first["published"] is True
        assert second["published"] is False and second["reason"] == "unchanged_inputs"

        # 3. 200-seed Monte Carlo progressive convergence evaluation across rollout checkpoints.
        mc_eval = run_monte_carlo_evaluation()

        output = {
            "draft_model_rejected": rejected_draft,
            "first_publish": first["published"],
            "second_publish_skipped_reason": second["reason"],
            "input_fingerprint": first["input_fingerprint"][:16],
            "horizon_1h_quantiles": {
                k: first["forecast"]["points"][0][k] for k in ("p10", "p50", "p90")
            },
            "monte_carlo_200_seed_eval": mc_eval,
        }
        print("=== Guarded Forecast Publisher & 200-Seed Rollout Convergence ===")
        print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
