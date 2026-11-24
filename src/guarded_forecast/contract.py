"""Model contract verification and deterministic SHA-256 input fingerprinting."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Set, Tuple, Union

MODEL_KIND = "linear_quantile_capacity_v1"
REQUIRED_QUANTILES: Tuple[str, ...] = ("p10", "p25", "p50", "p75", "p90")
MIN_VALIDATION_SAMPLES = 30
MIN_COVERAGE = 0.75
MAX_COVERAGE = 1.0


class ForecastModelError(ValueError):
    """Raised when a model artifact is malformed or fails its offline validation gate."""


def canonical_json(value: Any) -> str:
    """Serialize data deterministically for content hashing."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def require_finite_number(value: Any, *, field: str) -> float:
    """Validate that a contract numeric field is a finite real number (not bool)."""
    if isinstance(value, bool):
        raise ForecastModelError(f"{field} must be numeric, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ForecastModelError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ForecastModelError(f"{field} must be finite")
    return number


def load_validated_model(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate an approved quantile regression capacity model artifact.

    Fails closed unless the artifact declares `review_status == 'approved'`, meets
    minimum backtest sample and p10-p90 coverage thresholds, defines finite weights
    only for declared features, and specifies unique positive forecast horizons.
    """
    raw_text = Path(path).read_text(encoding="utf-8")
    try:
        artifact = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ForecastModelError(f"model artifact is not valid JSON: {exc}") from exc

    if not isinstance(artifact, dict):
        raise ForecastModelError("model artifact must be a JSON object")
    if artifact.get("kind") != MODEL_KIND:
        raise ForecastModelError(f"unsupported model kind: {artifact.get('kind')!r}")
    if artifact.get("review_status") != "approved":
        raise ForecastModelError("model artifact has not been approved for release")

    model_id = artifact.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ForecastModelError("model_id is required")
    model_version = artifact.get("model_version")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ForecastModelError("model_version is required")

    validation = artifact.get("validation")
    if not isinstance(validation, dict):
        raise ForecastModelError("validation metadata is required")

    sample_raw = validation.get("sample_count", 0)
    if isinstance(sample_raw, bool) or not isinstance(sample_raw, int):
        raise ForecastModelError("validation.sample_count must be an integer")
    if sample_raw < MIN_VALIDATION_SAMPLES:
        raise ForecastModelError(
            f"validation sample count {sample_raw} below minimum {MIN_VALIDATION_SAMPLES}"
        )

    coverage = require_finite_number(
        validation.get("p10_p90_coverage"),
        field="validation.p10_p90_coverage",
    )
    if not (MIN_COVERAGE <= coverage <= MAX_COVERAGE):
        raise ForecastModelError(
            f"validation.p10_p90_coverage {coverage} outside [{MIN_COVERAGE}, {MAX_COVERAGE}]"
        )

    feature_defaults = artifact.get("feature_defaults")
    if not isinstance(feature_defaults, dict) or not feature_defaults:
        raise ForecastModelError("feature_defaults must be a non-empty object")
    for fname, fdefault in feature_defaults.items():
        if not isinstance(fname, str) or not fname.strip():
            raise ForecastModelError("feature names must be non-empty strings")
        require_finite_number(fdefault, field=f"feature_defaults.{fname}")

    horizons = artifact.get("horizons")
    if not isinstance(horizons, list) or not horizons:
        raise ForecastModelError("horizons must be a non-empty list")

    seen_hours: Set[int] = set()
    for idx, row in enumerate(horizons):
        if not isinstance(row, dict):
            raise ForecastModelError(f"horizons[{idx}] must be an object")
        hours_raw = row.get("hours")
        if isinstance(hours_raw, bool) or not isinstance(hours_raw, int) or hours_raw <= 0:
            raise ForecastModelError("horizon hours must be unique positive integers")
        if hours_raw in seen_hours:
            raise ForecastModelError(f"duplicate horizon hours: {hours_raw}")
        seen_hours.add(hours_raw)

        coefficients = row.get("coefficients")
        if not isinstance(coefficients, dict):
            raise ForecastModelError(f"horizon {hours_raw}h coefficients must be an object")

        for quantile in REQUIRED_QUANTILES:
            q_row = coefficients.get(quantile)
            if not isinstance(q_row, dict):
                raise ForecastModelError(f"missing coefficients for {hours_raw}h.{quantile}")
            require_finite_number(
                q_row.get("intercept"),
                field=f"horizons.{hours_raw}.{quantile}.intercept",
            )
            weights = q_row.get("weights", {})
            if not isinstance(weights, dict):
                raise ForecastModelError(
                    f"horizons.{hours_raw}.{quantile}.weights must be an object"
                )
            for feat_name, weight_val in weights.items():
                if feat_name not in feature_defaults:
                    raise ForecastModelError(
                        f"horizons.{hours_raw}.{quantile} references undeclared feature {feat_name!r}"
                    )
                require_finite_number(
                    weight_val,
                    field=f"horizons.{hours_raw}.{quantile}.weights.{feat_name}",
                )

    return artifact


def input_fingerprint(snapshot: Mapping[str, Any], model: Mapping[str, Any]) -> str:
    """Compute a stable SHA-256 fingerprint over model identity and canonical snapshot inputs."""
    envelope = {
        "model_id": model["model_id"],
        "model_version": model["model_version"],
        "inputs": snapshot,
    }
    return hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()
