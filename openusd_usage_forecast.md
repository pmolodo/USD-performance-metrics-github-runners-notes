# OpenUSD GitHub Actions Trigger Forecast

Generated: 2026-10-01

## Forecast basis

The forecast uses inferred source-trigger events from cached GitHub Actions
runs. Runs sharing a source identity within five minutes count once. Reruns
are clustered separately and included in every forecast value.

The current release-cycle rate rounds to **67 triggering
events including reruns** per 30.4375 days.
The forecast starts at the least-squares line's value at the current
release boundary. Its linear slopes do not compound exponentially.

Partial release cycles are excluded from the fit because their event counts
are incomplete.

The fitted boundary is **59.8 events/month**, and the
observed slope is **11.7 events/month per release**.

| Scenario | Observed slope retained | Increase per release | Five-year events including reruns |
|---|---:|---:|---:|
| Low | 25% | +2.9 | 5,435 |
| Mid | 50% | +5.9 | 7,279 |
| High | 100% | +11.7 | 10,968 |

## Release-cycle seasonality

The recent burst is modeled as release-cycle seasonality rather than a
permanent increase in the trend. Each projected release cycle has three
equal 30.4375-day phases. Phase values are normalized within every cycle
so seasonality changes timing and peak capacity, not the scenario total.

| Release phase | Multiplier | Events/month at the current level |
|---|---:|---:|
| Early | 0.5x | 34 |
| Middle | 0.9x | 60 |
| Late | 1.6x | 108 |

The cycle average is the cost-planning quantity. The late-cycle value
is the runner-capacity quantity.

## Annual totals

| Year | Low incl. reruns | Mid incl. reruns | High incl. reruns |
|---|---:|---:|---:|
| Year 1 | 806 | 894 | 1,069 |
| Year 2 | 946 | 1,175 | 1,631 |
| Year 3 | 1,087 | 1,456 | 2,194 |
| Year 4 | 1,227 | 1,737 | 2,756 |
| Year 5 | 1,368 | 2,018 | 3,318 |

## Observed release-cycle rates

| Release cycle | Coverage | Events including reruns/month |
|---|---|---:|
| v25.08 | partial | 0.6 |
| v25.11 | partial | 35.5 |
| v26.03 | complete | 26.7 |
| v26.05 | complete | 39.8 |
| v26.08 | complete | 35.4 |
| post-v26.08 | in_progress | 67.2 |

## Generated forecast artifacts

- `openusd_triggering_events_forecast.csv`: 20 projected release cycles, with monthly rates and three-month totals
- `openusd_triggering_events_per_month_by_release_forecast.svg`: rates including reruns by actual and projected release

The SVG shows actual per-release monthly rates, a least-squares line over
the complete and in-progress cycles, and 20 projected release-cycle
averages. Inner-release monthly seasonality is intentionally omitted.

## Limitations

- The scenario slopes and phase multipliers are planning assumptions
  grounded in the available Actions observations, not statistically stable
  estimates from many release cycles.
- Refit the level and reconsider the slopes after each completed release.
- Trigger forecasts do not predict another workflow's duration. Apply
  platform-specific job times when converting events into runner minutes.
- Workflow configuration and triggering rules can change.

## Data coverage and reproducibility

- Repository ID: `58168143`
- Repository: `PixarAnimationStudios/OpenUSD`
- Cached Actions range ends: 2026-10-02 (exclusive)
- Complete monthly Actions observations: 13
- Cached workflows/runs/jobs: 4/579/5799

```bash
./run_openusd_usage.sh report
```
