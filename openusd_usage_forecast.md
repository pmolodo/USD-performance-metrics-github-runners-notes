# OpenUSD Pull-Request Activity and GitHub Actions Forecast

Generated: 2026-09-16

## Data coverage

- Repository ID: `58168143`
- Historical names: `PixarAnimationStudios/USD` and `PixarAnimationStudios/OpenUSD`
- Pull-request event source: https://sql-clickhouse.clickhouse.com/
- Cached PR-event range: 2021-09-16 through 2026-09-17 (exclusive)
- Complete months modeled: 2021-10-01 through 2026-08-31
- Complete monthly observations: 59
- `BuildUSD` workflow record created: 2024-09-09
- Cached `BuildUSD` run months: 2025-07-01 through 2026-09-01
- Cached workflows/runs/jobs: 4/452/2592
- Cached PR timelines/events: 1195/18558

The stable five-year demand proxy counts PR openings and reopenings. GitHub
Actions also uses `synchronize` as a default `pull_request` activity, but
GitHub's historical APIs do not expose exact push timestamps for that event.
The cached timeline commits use author/committer dates and are therefore not
treated as exact workflow triggers. Calibration against actual runs absorbs
the average effect of synchronizations and reruns during the overlap period.

## Model comparison

Models were compared over the latest 24 observations using expanding-window,
one-month-ahead validation. A long-horizon guardrail rejects a model when its
Year 5 forecast is below 25% or above 400% of the recent annualized mean.
The lowest-MAE model that passes the guardrail is selected.

| Model | Validation MAE | Validation RMSE | Guardrail |
|---|---:|---:|---|
| quadratic | 3.16 | 4.35 | reject |
| trailing-24-month mean | 11.56 | 12.67 | pass |
| constant | 11.66 | 13.41 | pass |
| exponential | 11.97 | 12.82 | reject |
| linear | 13.59 | 14.22 | reject |

Selected model: **trailing-24-month mean**.

## Five-year trigger forecast

| Forecast year | PR-trigger proxy | Calibrated proxy runs | Flat actual-run baseline |
|---|---:|---:|---:|
| Year 1 | 136 | 653 | 325 |
| Year 2 | 136 | 653 | 325 |
| Year 3 | 136 | 653 | 325 |
| Year 4 | 136 | 653 | 325 |
| Year 5 | 136 | 653 | 325 |

## Runner-minute baseline

The most defensible near-term capacity baseline is the measured runner
time from the latest 12 complete months. The five-year column holds that
annual workload flat; it includes both PR and push workflow runs.

| Runner label | Cached jobs | Cached minutes | Latest 12-month minutes | Five-year flat minutes |
|---|---:|---:|---:|---:|
| `["enterprise-linux-x64-t4gpu-4core-16vram-28ram-176ssd"]` | 244 | 389 | 382 | 1,910 |
| `["macos-15"]` | 460 | 25,238 | 22,280 | 111,400 |
| `["ubuntu-22.04"]` | 1,428 | 50,937 | 45,167 | 225,834 |
| `["windows-2022"]` | 460 | 32,747 | 29,846 | 149,230 |

## Proxy validation against actual BuildUSD runs

Across 14 overlapping complete months, the proxy counted
73 trigger events and GitHub recorded 352
pull-request workflow runs. The resulting calibration factor is
**4.822 actual runs per proxy trigger**.

The monthly proxy-to-run correlation is **-0.530**,
and its calibrated monthly MAE is **23.2 runs**. A weak or
negative correlation means the calibrated proxy is not suitable as a
standalone point estimate. The flat baseline annualizes the latest 12
complete months of actual workflow data to **325 runs**.

| Month | Trigger proxy | Actual PR workflow runs |
|---|---:|---:|
| 2025-07 | 21 | 1 |
| 2025-08 | 6 | 26 |
| 2025-09 | 6 | 24 |
| 2025-10 | 11 | 33 |
| 2025-11 | 2 | 10 |
| 2025-12 | 10 | 29 |
| 2026-01 | 6 | 17 |
| 2026-02 | 6 | 26 |
| 2026-03 | 3 | 24 |
| 2026-04 | 2 | 25 |
| 2026-05 | 0 | 20 |
| 2026-06 | 0 | 29 |
| 2026-07 | 0 | 43 |
| 2026-08 | 0 | 45 |

## Limitations

- Pull-request activity predicts workflow starts, not runner duration.
- Openings/reopenings are a coarse proxy for synchronization-heavy PRs;
  inspect the monthly validation table before using the point forecast.
- Timeline commit events do not preserve exact push times or how commits were
  grouped into pushes, so they are cached but excluded from trigger counts.
- The workflow excludes changes limited to `.github/workflows/**`.
- Workflow configuration, job count, and runner speed can change.
- Five-year extrapolation is substantially more uncertain than the
  one-month validation used to select the model.
- Cost forecasts should multiply predicted runs by measured job duration
  from the cached Actions jobs, not by an assumed one-hour runtime.

## Database and reproducibility

The embedded Dolt database is `.cache/openusd_usage_dolt`. It is ignored
by Git, while its internal Dolt commits preserve each completed import
as a queryable data snapshot. Python dependencies are locked with `uv`.
No Dolt SQL server is required or started.

```bash
./run_openusd_usage.sh sync-all --start-date 2021-09-16
./run_openusd_usage.sh report
dolt -C .cache/openusd_usage_dolt log --oneline
```

Dolt 2.3 supports Git-backed database remotes. This database is pushed
to the existing GitHub repository through its separate `refs/dolt/data`
ref, so the database history coexists with and does not alter Git
`main`. The configured remote is:

```text
git+ssh://git@github.com/./pmolodo/USD-performance-metrics-github-runners-notes.git
```

After a successful sync, publish the new Dolt commits with:

```bash
dolt -C .cache/openusd_usage_dolt push
```

To reconstruct the cache in a fresh source checkout:

```bash
dolt clone git@github.com:pmolodo/USD-performance-metrics-github-runners-notes.git .cache/openusd_usage_dolt
```

See Dolt's [Git remote URL implementation][dolt-git-remote] and
[Git remote integration tests][dolt-git-remote-tests].

[dolt-git-remote]: https://github.com/dolthub/dolt/blob/main/go/libraries/doltcore/env/git_remote_url.go
[dolt-git-remote-tests]: https://github.com/dolthub/dolt/blob/main/integration-tests/bats/remotes-git.bats
