# guarded-forecast-publisher

A Python 3.11+ verification-gated quantile publisher (CI-verified on Ubuntu 3.11 and 3.12; compatible with 3.9+) and Empirical Bayes partial-rollout estimator for multi-region capacity and canary telemetry.

---

## Why This Exists

Two quiet bugs show up whenever operational forecasting or rollout telemetry ships without hard gates:

1. **Unreviewed models slip into output files:** If data ingestion and forecast publishing live in the same loop, a draft quantile regression artifact or crossed quantile prediction (`p10 > p90`) silently overwrites production JSON whenever a new metric arrives.
2. **Fast shards bias early rollout averages:** During progressive canary rollouts, low-load or isolated regions finish reporting long before high-volume core clusters. Computing a naive unweighted average over the first 5% to 25% of shards exaggerates regressions or hides baseline shifts until late in the rollout.

`guarded-forecast-publisher` addresses both problems with two decoupled primitives:

- **Contract-gated atomic publishing:** Separates telemetry snapshots from model approval (`review_status == "approved"`, minimum backtest sample count and `p10-p90` coverage, non-crossing quantile enforcement, and SHA-256 `input_fingerprint` skip-when-unchanged before `os.replace()`).
- **Empirical Bayes rollout shrinkage (`lambda = n / (n + k)`):** Anchors low-sample or unreported regions toward the fleet baseline prior, models standard error floor bounds (`max(se_floor, se_scale / sqrt(n))`), and emits Monte Carlo 95% credible intervals verified by a progressive convergence backtest.

## Origin & Sanitization

These two patterns were extracted and rewritten from private probabilistic forecasting and real-time aggregation pipelines. All proprietary domain equations, lag features, regional priors, historical training datasets, and external data scrapers remain strictly private; the public toolkit retains only the verification gate, idempotent SHA-256 publisher, Empirical Bayes shrinkage formulas, and synthetic cluster-capacity replay harness.

## What It Intentionally Does Not Do

This repository is intentionally a narrow gate and statistical aggregation package (`numpy` only).

It does **not**:

- fit or train quantile regression models online (model artifacts must be validated offline first)
- fetch external metrics over HTTP or mutate cluster autoscalers
- store multi-year time-series metrics (writes immutable JSON forecast snapshots and compact fingerprint state)

---

## Architecture

```text
Telemetry Snapshot                      Versioned Model Artifact
(cluster_id + metrics)                  (review_status, validation, horizons)
         |                                           |
         +-------------------+-----------------------+
                             |
                             v
               load_validated_model()   [Rejects draft / low-sample / low-coverage]
                             |
                             v
               input_fingerprint()      [SHA-256 match -> skip disk rewrite]
                             |
                             v
               evaluate_quantiles()     [Rejects p10..p90 quantile crossing]
                             |
                             v
               atomic_write_json()      [temp file + fsync + os.replace]

Partial Canary Shards -----> estimate_empirical_bayes_rollout()
(reported_fraction, N)       |-- Filter min fraction & min sample count
                             |-- Shrink low-N regions: lambda = n / (n + k)
                             +-- Monte Carlo draws -> 95% CI & SLO breach prob
```

## Local Quickstart

Requires **Python 3.11+** (CI-verified on Ubuntu 3.11 and 3.12; compatible with 3.9+, `numpy` + `pytest` for tests).

```bash
python -m pip install -e ".[dev]"
pytest -v
python scripts/run_replay_demo.py
```

## Measured Replay Convergence Results

Output from `python scripts/run_replay_demo.py` (`true_fleet_delta = 0.008308` across 60 shards with skewed fast-reporting early clusters):

| Rollout Checkpoint | Reported Shards | Raw Mean | Raw Abs Error | EB Shrunk Mean | EB Abs Error | 95% Credible Interval |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `5%` | `3` | `0.06` | `0.051692` | `0.005976` | `0.002332` | `[-0.008834, 0.023943]` |
| `25%` | `15` | `0.0264` | `0.018092` | `0.006441` | `0.001867` | `[-0.005723, 0.020357]` |
| `50%` | `30` | `0.0152` | `0.006892` | `0.006273` | `0.002035` | `[-0.001197, 0.016206]` |
| `100%` | `60` | `0.0096` | `0.001292` | `0.005993` | `0.002315` | `[0.003042, 0.00961]` |

At the `5%` skewed canary rollout checkpoint, Empirical Bayes shrinkage reduces absolute estimation error by ~22.2x (`0.002332` vs `0.051692`).

## License

MIT - see [LICENSE](LICENSE).
