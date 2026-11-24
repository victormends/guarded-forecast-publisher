"""Runnable local demo: contract-gated quantile publishing and canary rollout replay."""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from guarded_forecast import (
    MODEL_KIND,
    ForecastModelError,
    ShardObservation,
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

        # 3. Skewed progressive canary rollout replay: low-latency canary shards report before heavy regions.
        events: List[ShardObservation] = []
        # Fast anomalous region (small capacity share, reports first with +0.060 latency delta)
        for i in range(6):
            events.append(
                ShardObservation(
                    region_id="region-canary-fast",
                    shard_id=f"c-{i}",
                    reported_fraction=1.0,
                    sample_size=450,
                    observed_delta=0.058 + (i % 3) * 0.002,
                )
            )
        # Larger slower regions report later close to baseline (+0.004 delta)
        for reg in ("region-core-a", "region-core-b", "region-core-c"):
            for i in range(18):
                events.append(
                    ShardObservation(
                        region_id=reg,
                        shard_id=f"{reg}-{i}",
                        reported_fraction=1.0,
                        sample_size=600,
                        observed_delta=0.003 + (i % 3) * 0.001,
                    )
                )

        # True weighted fleet delta across regions (weights: fast=1, core-a=4, core-b=4, core-c=4)
        weights = {"region-canary-fast": 1.0, "region-core-a": 4.0, "region-core-b": 4.0, "region-core-c": 4.0}
        true_delta = round(
            (1.0 * 0.060 + 4.0 * 0.004 + 4.0 * 0.004 + 4.0 * 0.004) / 13.0,
            6,
        )
        replay = run_convergence_backtest(
            events,
            true_fleet_delta=true_delta,
            fleet_prior_delta=0.005,
            checkpoints=(0.05, 0.25, 0.50, 1.0),
            prior_strength_k=10.0,
            region_weights=weights,
            seed=20261124,
        )

        output = {
            "draft_model_rejected": rejected_draft,
            "first_publish": first["published"],
            "second_publish_skipped_reason": second["reason"],
            "input_fingerprint": first["input_fingerprint"][:16],
            "horizon_1h_quantiles": {
                k: first["forecast"]["points"][0][k] for k in ("p10", "p50", "p90")
            },
            "true_fleet_delta": true_delta,
            "rollout_convergence": [
                {
                    "checkpoint_pct": int(row.checkpoint_fraction * 100),
                    "reported_shards": row.reported_shards,
                    "raw_mean": row.raw_mean,
                    "raw_abs_err": row.raw_abs_error,
                    "eb_shrunk_mean": row.shrunk_mean,
                    "eb_abs_err": row.shrunk_abs_error,
                    "ci95": [row.ci95_low, row.ci95_high],
                }
                for row in replay
            ],
        }
        print("=== Guarded Forecast Publisher & Rollout Convergence Demo ===")
        print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
