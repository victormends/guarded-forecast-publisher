"""Tests for model artifact validation gates, non-crossing quantiles, and atomic publishing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from guarded_forecast.contract import MODEL_KIND, ForecastModelError, load_validated_model
from guarded_forecast.publisher import publish_if_inputs_changed


def _base_model() -> Dict[str, Any]:
    return {
        "kind": MODEL_KIND,
        "model_id": "cap-model",
        "model_version": "1.2.0",
        "review_status": "approved",
        "validation": {
            "sample_count": 120,
            "p10_p90_coverage": 0.82,
        },
        "feature_defaults": {
            "cpu_util_pct": 40.0,
            "queue_depth": 10.0,
        },
        "horizons": [
            {
                "hours": 4,
                "coefficients": {
                    "p10": {"intercept": 4.0, "weights": {"cpu_util_pct": 0.6}},
                    "p25": {"intercept": 6.0, "weights": {"cpu_util_pct": 0.7}},
                    "p50": {"intercept": 8.0, "weights": {"cpu_util_pct": 0.8, "queue_depth": 0.1}},
                    "p75": {"intercept": 10.0, "weights": {"cpu_util_pct": 0.9, "queue_depth": 0.1}},
                    "p90": {"intercept": 12.0, "weights": {"cpu_util_pct": 1.0, "queue_depth": 0.2}},
                },
            }
        ],
    }


def test_contract_rejects_unapproved_undercovered_and_malformed_models(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"

    # 1. Valid model passes
    model_path.write_text(json.dumps(_base_model()), encoding="utf-8")
    loaded = load_validated_model(model_path)
    assert loaded["model_id"] == "cap-model"

    # 2. Unapproved status fails closed
    bad = _base_model()
    bad["review_status"] = "pending_review"
    model_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ForecastModelError, match="approved"):
        load_validated_model(model_path)

    # 3. Insufficient sample size fails closed
    bad = _base_model()
    bad["validation"]["sample_count"] = 12
    model_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ForecastModelError, match="below minimum"):
        load_validated_model(model_path)

    # 4. Out-of-bounds coverage fails closed
    bad = _base_model()
    bad["validation"]["p10_p90_coverage"] = 0.60
    model_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ForecastModelError, match="coverage"):
        load_validated_model(model_path)

    # 5. Bool disguised as numeric or unknown feature name fails closed
    bad = _base_model()
    bad["feature_defaults"]["cpu_util_pct"] = True
    model_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ForecastModelError, match="numeric"):
        load_validated_model(model_path)

    bad = _base_model()
    bad["horizons"][0]["coefficients"]["p50"]["weights"]["undeclared_feat"] = 1.0
    model_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ForecastModelError, match="undeclared feature"):
        load_validated_model(model_path)


def test_publish_if_inputs_changed_skips_unchanged_and_discloses_missing_features(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    out_path = tmp_path / "out" / "forecast.json"
    state_path = tmp_path / "state" / "state.json"
    model_path.write_text(json.dumps(_base_model()), encoding="utf-8")

    issued_at = datetime(2026, 11, 24, 10, 0, 0, tzinfo=timezone.utc)
    # queue_depth omitted intentionally -> defaults apply and missing_features discloses it.
    snapshot_v1 = {"cluster_id": "cluster-a", "metrics": {"cpu_util_pct": 50.0}}

    res1 = publish_if_inputs_changed(
        model_path=model_path,
        input_snapshot=snapshot_v1,
        output_path=out_path,
        state_path=state_path,
        issued_at=issued_at,
    )
    assert res1["published"] is True
    assert res1["forecast"]["missing_features"] == ["queue_depth"]
    assert res1["forecast"]["features"]["queue_depth"] == 10.0
    assert out_path.is_file()
    assert state_path.is_file()
    assert list(out_path.parent.glob("*.tmp")) == []

    # Same inputs + same model version -> idempotent skip
    res2 = publish_if_inputs_changed(
        model_path=model_path,
        input_snapshot=snapshot_v1,
        output_path=out_path,
        state_path=state_path,
        issued_at=issued_at,
    )
    assert res2["published"] is False
    assert res2["reason"] == "unchanged_inputs"
    assert res2["input_fingerprint"] == res1["input_fingerprint"]

    # Changed telemetry triggers a new atomic revision
    snapshot_v2 = {"cluster_id": "cluster-a", "metrics": {"cpu_util_pct": 75.0, "queue_depth": 22.0}}
    res3 = publish_if_inputs_changed(
        model_path=model_path,
        input_snapshot=snapshot_v2,
        output_path=out_path,
        state_path=state_path,
        issued_at=issued_at,
    )
    assert res3["published"] is True
    assert res3["input_fingerprint"] != res1["input_fingerprint"]
    assert res3["forecast"]["missing_features"] == []


def test_crossed_quantiles_or_failed_gate_leave_live_output_intact(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    out_path = tmp_path / "out" / "forecast.json"
    state_path = tmp_path / "state" / "state.json"
    model_path.write_text(json.dumps(_base_model()), encoding="utf-8")

    publish_if_inputs_changed(
        model_path=model_path,
        input_snapshot={"cluster_id": "cluster-a", "metrics": {"cpu_util_pct": 50.0}},
        output_path=out_path,
        state_path=state_path,
    )
    baseline_bytes = out_path.read_bytes()

    # Negative weights invert slope so p10 exceeds p90 at high utilization -> fails closed.
    crossed = _base_model()
    crossed["horizons"][0]["coefficients"]["p10"] = {"intercept": 50.0, "weights": {"cpu_util_pct": 1.0}}
    crossed["horizons"][0]["coefficients"]["p90"] = {"intercept": 10.0, "weights": {"cpu_util_pct": 0.1}}
    model_path.write_text(json.dumps(crossed), encoding="utf-8")

    with pytest.raises(ForecastModelError, match="crossed"):
        publish_if_inputs_changed(
            model_path=model_path,
            input_snapshot={"cluster_id": "cluster-a", "metrics": {"cpu_util_pct": 80.0}},
            output_path=out_path,
            state_path=state_path,
        )

    assert out_path.read_bytes() == baseline_bytes
    assert list(out_path.parent.glob("*.tmp")) == []
