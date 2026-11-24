"""Fail-closed forecast evaluator and idempotent atomic JSON publisher."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from .contract import (
    REQUIRED_QUANTILES,
    ForecastModelError,
    input_fingerprint,
    load_validated_model,
    require_finite_number,
)


def extract_features(
    snapshot: Mapping[str, Any],
    feature_defaults: Mapping[str, Any],
) -> Tuple[Dict[str, float], List[str]]:
    """Map telemetry snapshot fields onto declared model features with missing-field disclosure."""
    raw_metrics = snapshot.get("metrics")
    metrics: Mapping[str, Any] = raw_metrics if isinstance(raw_metrics, Mapping) else {}

    resolved: Dict[str, float] = {}
    missing_features: List[str] = []
    for name, default_val in feature_defaults.items():
        candidate = metrics.get(name)
        if candidate is None:
            missing_features.append(name)
            resolved[name] = require_finite_number(
                default_val, field=f"feature_defaults.{name}"
            )
            continue
        resolved[name] = require_finite_number(candidate, field=f"metrics.{name}")
    return resolved, sorted(missing_features)


def evaluate_quantiles(
    horizon_row: Mapping[str, Any],
    features: Mapping[str, float],
    *,
    round_digits: int = 4,
) -> Dict[str, float]:
    """Evaluate quantile outputs for one horizon and reject quantile crossing."""
    hours = int(horizon_row["hours"])
    coefficients = horizon_row["coefficients"]
    result: Dict[str, float] = {}

    for quantile in REQUIRED_QUANTILES:
        q_spec = coefficients[quantile]
        value = require_finite_number(
            q_spec["intercept"], field=f"{hours}h.{quantile}.intercept"
        )
        for feat_name, weight in q_spec.get("weights", {}).items():
            value += require_finite_number(
                weight, field=f"{hours}h.{quantile}.weights.{feat_name}"
            ) * float(features[feat_name])
        if not math_isfinite(value):
            raise ForecastModelError(f"non-finite prediction at {hours}h.{quantile}")
        result[quantile] = round(value, round_digits)

    ordered = [result[q] for q in REQUIRED_QUANTILES]
    if ordered != sorted(ordered):
        raise ForecastModelError(f"quantiles are crossed at {hours}h: {result}")
    return result


def math_isfinite(val: float) -> bool:
    import math

    return math.isfinite(val)


def issue_forecast(
    *,
    model: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    issued_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Evaluate all approved horizons into a structured capacity forecast payload."""
    ts = issued_at or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(timezone.utc)

    target_id = str(snapshot.get("cluster_id") or "").strip()
    if not target_id:
        raise ForecastModelError("snapshot.cluster_id is required")

    features, missing_features = extract_features(snapshot, model["feature_defaults"])
    points: List[Dict[str, Any]] = []
    for row in sorted(model["horizons"], key=lambda item: int(item["hours"])):
        hours = int(row["hours"])
        quantiles = evaluate_quantiles(row, features)
        valid_at = ts + timedelta(hours=hours)
        points.append({"hours": hours, "valid_at": valid_at.isoformat(), **quantiles})

    return {
        "schema_version": 1,
        "cluster_id": target_id,
        "issued_at": ts.isoformat(),
        "model": {
            "id": model["model_id"],
            "version": model["model_version"],
            "validation": dict(model["validation"]),
        },
        "features": features,
        "missing_features": missing_features,
        "points": points,
    }


def atomic_write_json(path: Union[str, Path], payload: Mapping[str, Any]) -> Path:
    """Write JSON atomically via unique temp file, flush, fsync, and os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    serialized = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(str(tmp_path), str(target))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return target


def _read_state_fingerprint(state_path: Path) -> Optional[str]:
    if not state_path.exists():
        return None
    try:
        parsed = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    fp = parsed.get("last_fingerprint")
    return str(fp) if fp else None


def publish_if_inputs_changed(
    *,
    model_path: Union[str, Path],
    input_snapshot: Mapping[str, Any],
    output_path: Union[str, Path],
    state_path: Union[str, Path],
    issued_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Validate the model gate, skip unchanged inputs via SHA-256, and publish atomically."""
    model = load_validated_model(model_path)
    fingerprint = input_fingerprint(input_snapshot, model)

    out_file = Path(output_path)
    state_file = Path(state_path)
    previous_fp = _read_state_fingerprint(state_file)
    if previous_fp == fingerprint and out_file.is_file():
        existing_payload = json.loads(out_file.read_text(encoding="utf-8"))
        return {
            "published": False,
            "reason": "unchanged_inputs",
            "input_fingerprint": fingerprint,
            "output_path": str(out_file),
            "forecast": existing_payload,
        }

    forecast = issue_forecast(model=model, snapshot=input_snapshot, issued_at=issued_at)
    forecast["input_fingerprint"] = fingerprint

    atomic_write_json(out_file, forecast)
    atomic_write_json(
        state_file,
        {
            "last_fingerprint": fingerprint,
            "published_at": forecast["issued_at"],
            "model_id": model["model_id"],
            "model_version": model["model_version"],
            "cluster_id": forecast["cluster_id"],
        },
    )
    return {
        "published": True,
        "reason": "published",
        "input_fingerprint": fingerprint,
        "output_path": str(out_file),
        "forecast": forecast,
    }
