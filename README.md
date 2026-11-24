# guarded-forecast-publisher

A Python 3.11+ verification-gated quantile publisher (CI-verified on Ubuntu 3.11 and 3.12) and Empirical Bayes partial-rollout estimator for multi-region capacity and canary telemetry.

---

## Why This Exists

Two quiet bugs show up whenever operational forecasting or rollout telemetry ships without hard gates:

1. **Unreviewed models slip into output files:** If data ingestion and forecast publishing live in the same loop, a draft quantile regression artifact or crossed quantile prediction (`p10 > p90`) silently overwrites production JSON whenever a new metric arrives.
2. **Fast shards bias early rollout averages:** During progressive canary rollouts, low-load or isolated regions finish reporting long before high-volume core clusters. Computing a naive unweighted average over the first 5% to 25% of shards exaggerates regressions or hides baseline shifts until late in the rollout.

`guarded-forecast-publisher` addresses both problems with two decoupled primitives:

- **Contract-gated atomic publishing:** Separates telemetry snapshots from model approval (`review_status == "approved"`, minimum backtest sample count and `p10-p90` coverage, non-crossing quantile enforcement, and SHA-256 `input_fingerprint` skip-when-unchanged before `os.replace()`).
- **Normal-Normal conjugate Empirical Bayes shrinkage:** Given `prior_sigma` (`tau_0`) and `within_shard_sigma` (`sigma_obs`), weights each region by `lambda_r = tau_0^2 / (tau_0^2 + (sigma_obs^2 / n_r) * fpc)` toward `fleet_prior_delta` with posterior variance `(1 - lambda_r) * tau_0^2`, plus optional finite-population correction (`region_total_shards`, driving `lambda_r -> 1.0` and `SE -> 0` at 100% reporting) and a legacy `lambda = n / (n + k)` mode when sigma priors are omitted.

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
                             |-- Conjugate EB: lambda = tau0^2 / (tau0^2 + sigma_obs^2/n * fpc)
                             +-- Monte Carlo draws -> calibrated 95% CI & SLO breach prob
```

## Local Quickstart

Requires **Python 3.11+** (CI-verified on Ubuntu 3.11 and 3.12, `numpy` + `pytest` for tests).

```bash
python -m pip install -e ".[dev]"
pytest -v
python scripts/run_replay_demo.py
```

## Measured 200-Seed Hierarchical Monte Carlo Convergence Results

Output from `python scripts/run_replay_demo.py` across `seeds 1..200` (`fleet_prior_delta = 0.005`, `prior_sigma = 0.012`, `within_shard_sigma = 0.018`, `60` shards across `4` regions with skewed fast-reporting early canary shards):

| Rollout Checkpoint | Reported Shards | Raw MAE | EB Shrunk MAE | Raw Median AE | EB Median AE | Empirical 95% CI Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `5%` | `3` | `0.012506` | `0.005259` | `0.010786` | `0.004470` | `0.9450` (`189/200`) |
| `25%` | `15` | `0.010966` | `0.005225` | `0.009735` | `0.004426` | `0.9450` (`189/200`) |
| `50%` | `30` | `0.007067` | `0.004493` | `0.006320` | `0.003600` | `0.9500` (`190/200`) |
| `100%` | `60` | `0.002686` | `0.001846` | `0.002152` | `0.001761` | `0.9650` (`193/200`) |

### Conjugate Posterior & Finite-Population Behavior Across Full Rollout

Under the Normal-Normal conjugate Empirical Bayes estimator (`lambda_r = tau_0^2 / (tau_0^2 + sigma_obs^2 / n_r)` with posterior standard error `sqrt((1 - lambda_r) * tau_0^2)`), EB shrinkage improves accuracy over the raw sample mean at every rollout stage (`5%` through `100%`, cutting MAE from `0.012506` to `0.005259` at `5%` and from `0.002686` to `0.001846` at `100%`) while holding empirical 95% credible interval coverage calibrated (`94.5%` to `96.5%` across all four checkpoints). When finite-population convergence to the realized cohort mean is desired instead of super-population inference, passing `region_total_shards` applies a finite-population correction (`fpc = max(0, 1 - n_r / N_tot)`), driving `lambda_r = 1.0`, `shrunk_mean = raw_mean`, and `standard_error = 0.0` at `100%` reporting.

## License

MIT - see [LICENSE](LICENSE).
