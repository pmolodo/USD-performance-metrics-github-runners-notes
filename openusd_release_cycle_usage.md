# OpenUSD GitHub Actions Usage by Release Cycle

OpenUSD usage is grouped into development cycles bounded by publication
timestamps from the official [GitHub releases][openusd-releases]. Each completed
cycle is named for its target release. For example, `v26.03` covers the interval
from publication of `v25.11` through publication of `v26.03`. `post-v26.08` is
the observed portion of the current cycle and is not a completed release
interval.

Monthly rates use exactly 30.4375 days. Totals and normalized rates are both
provided so that cycles of different lengths can be compared.

Only cadence tags matching `vYY.MM` define cycle boundaries. Patch and preview
tags such as `v25.05.01` and `v25.02a` remain cached as raw release metadata but
do not start new development cycles.

## Summary scope

Trigger counts infer one source event across workflows sharing the same commit
identity within five minutes. Rerun events are also reported separately and are
clustered across workflows when they occur together.

Runner-minute summaries include only these exact `BuildUSD` jobs:

- `Linux` on `ubuntu-22.04`
- `macOS` on `macos-15`
- `Windows` on `windows-2022`

Wasm, Wasm64, GPU tests, validation, packaging matrices, and other workflows
are excluded from summary minutes. Their runs, jobs, steps, and durations remain
available in the raw Dolt tables.

## Release-cycle results

| Release cycle | Coverage | Days | 30.4375-day months | Source events | Source events/month | Events plus reruns/month |
|---|---|---:|---:|---:|---:|---:|
| v25.08 | partial | 93.84 | 3.083 | 2 | 0.6 | 0.6 |
| v25.11 | partial | 84.97 | 2.792 | 94 | 33.7 | 35.5 |
| v26.03 | complete, long | 123.17 | 4.047 | 105 | 25.9 | 26.7 |
| v26.05 | complete | 58.85 | 1.933 | 65 | 33.6 | 39.8 |
| v26.08 | complete | 86.81 | 2.852 | 95 | 33.3 | 35.4 |
| post-v26.08 | in progress | 73.40 | 2.411 | 157 | 65.1 | 67.2 |

`v26.03` is the significantly long completed cycle. Its 123.17-day interval is
more than 25% longer than the median interval between official cadence releases.

The three complete release cycles have source-event rates of 25.9, 33.6, and
33.3 per standardized month. The available complete-cycle sample therefore does
not yet establish a repeating peak-and-dip pattern tied to the release cadence.
The current cycle's 65.1 source events per month is substantially higher, but it
is an unfinished interval and should not be treated as a completed-cycle result.

## Core runner-minute rates

| Release cycle | Linux x64 minutes/month | macOS minutes/month | Windows minutes/month |
|---|---:|---:|---:|
| v25.08 | 37 | 25 | 48 |
| v25.11 | 1,914 | 1,880 | 2,611 |
| v26.03 | 1,245 | 1,416 | 1,944 |
| v26.05 | 2,163 | 1,970 | 4,041 |
| v26.08 | 2,110 | 2,092 | 2,670 |
| post-v26.08 | 3,495 | 3,125 | 1,730 |

The current interval shows higher Linux x64 and macOS demand and lower Windows
demand than the completed `v26.08` cycle. Because the interval is still in
progress, these values are rates based on elapsed time rather than final totals.

## Generated data and charts

- `openusd_triggering_events_by_release.csv`: release boundaries, duration,
  coverage, raw totals, normalized rates, and per-source-event platform minutes
- `openusd_triggering_events_by_release.svg`: triggering-event totals
- `openusd_triggering_events_per_month_by_release.svg`: normalized trigger rates
- `openusd_runner_minutes_by_release.svg`: core platform runner-minute totals
- `openusd_runner_minutes_per_month_by_release.svg`: normalized platform rates

The first two cycles are visibly marked as partial because complete all-workflow
collection begins on 2025-09-01. The current cycle is marked as in progress.

[openusd-releases]: https://github.com/PixarAnimationStudios/OpenUSD/releases
