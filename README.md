# guarded-forecast-publisher

A Python 3.11+ verification-gated quantile publisher (CI-verified on Ubuntu 3.11 and 3.12) and Empirical Bayes partial-rollout estimator for multi-region capacity and canary telemetry.

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

Requires **Python 3.11+** (CI-verified on Ubuntu 3.11 and 3.12, `numpy` + `pytest` for tests).

```bash
python -m pip install -e ".[dev]"
pytest -v
python scripts/run_replay_demo.py
```

## Measured 50-Seed Monte Carlo Convergence Results

Output from `python scripts/run_replay_demo.py` across `seeds 1..50` (`true_fleet_delta = 0.008308` across `60` shards with skewed fast-reporting early clusters):

| Rollout Checkpoint | Reported Shards | Raw MAE | EB Shrunk MAE | Raw Median AE | EB Median AE | Empirical 95% CI Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `5%` | `3` | `0.051262` | `0.002339` | `0.052918` | `0.002310` | `1.0000` (`50/50`) |
| `25%` | `15` | `0.018327` | `0.001778` | `0.018217` | `0.001804` | `1.0000` (`50/50`) |
| `50%` | `30` | `0.006475` | `0.002238` | `0.006451` | `0.002240` | `1.0000` (`50/50`) |
| `100%` | `60` | `0.001663` | `0.002309` | `0.001368` | `0.002377` | `0.8000` (`40/50`) |

### Bias-Variance Trade-Off at `100%` Completion

Empirical Bayes shrinkage improves estimation accuracy sharply during early partial rollouts (`5%` to `50%`), cutting Mean Absolute Error by ~21.9x at `5%` (`0.002339` vs `0.051262`) and ~10.3x at `25%` (`0.001778` vs `0.018327`) while preserving `50/50` (`100%`) credible interval coverage. Once `100%` of clusters have reported, however, continued shrinkage toward `fleet_prior_delta` introduces residual bias (`EB MAE = 0.002309` vs `Raw MAE = 0.001663`, dropping empirical 95% CI coverage to `40/50` / `80%`) relative to the full-fleet raw sample mean. Empirical Bayes shrinkage is therefore designed strictly for early partial rollouts (`5%` to `50%`) before switching to the raw unweighted fleet mean near completion.

## License

MIT - see [LICENSE](LICENSE).
