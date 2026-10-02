# Existing OpenUSD workflow: historical costs and projections

Analysis date: 2026-10-02. Currency: USD.

## Headline: BuildUSD, not the custom 16-configuration matrix

Workflow reference: [BuildUSD at the latest cached build commit](https://github.com/PixarAnimationStudios/OpenUSD/blob/240e3ddef8d79a5800490c85ca4a7741397f8516/.github/workflows/buildusd.yml).

Retained execution window: **2025-07-31 19:11:16 through 2026-10-01 17:58:13.104208 UTC**.
This is a public-data list-price estimate, not Pixar's invoice. Discounts,
credits, sponsorship, enterprise seats, and account-level storage settings
are not visible. Expired/deleted runs outside the cache cannot be reconstructed.

| Historical BuildUSD cost | Public repository | Private, resource-matched proxy | Private, same labels and recorded times |
|---|---:|---:|---:|
| Runner compute | $19.03 | $3,470.29 | $2,659.61 |
| Estimated artifact storage | $0.00 | $260.94 | $260.94 |
| Total of modeled items | **$19.03** | **$3,731.24** | **$2,920.56** |

The public compute charge is entirely the Linux GPU-test runner. Standard
Linux, Windows, macOS, validation, and Wasm jobs are free in the public case.
Private cases apply no included minutes or artifact allowance, as requested.
They are marginal costs with the account allowance exhausted, not a statement
that GitHub private plans have no allowance.

## What 'equivalent paid runners' means

[Standard runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
give public Linux/Windows labels 4 vCPUs and 16 GB RAM, versus 2 vCPUs and
8 GB in private repositories. macOS retains the same standard shape.
The main proxy prices recorded times at paid 4-core Linux/Windows rates.
This would require runner routing changes, not an untouched YAML file.

Important: the [larger-runner reference](https://docs.github.com/en/actions/reference/runners/larger-runners)
limits paid 4-core Windows runners to Windows Server 2025 or Windows 11,
not the workflow's Windows Server 2022. Thus the 4-core Windows amount is
a CPU/RAM-matched price proxy, not an exact purchasable 2022 equivalent.
The same-label column leaves runner labels unchanged, but reuses the public
durations on smaller Linux/Windows VMs. It is a time-held-constant comparison,
not a validated budget: runtimes, failures, and cache behavior can change.
No unsupported 2x runtime assumption is silently applied.

## Rates and historical accounting

| Runner | Before 2026, $/minute | Since 2026-01-01, $/minute |
|---|---:|---:|
| Linux x64, paid 4-core proxy | 0.016 | 0.012 |
| Windows x64, paid 4-core proxy | 0.032 | 0.022 |
| macOS standard | 0.080 | 0.062 |
| Linux T4 GPU, 4-core | 0.070 | 0.052 |
| Linux x64, private standard 2-core | 0.008 | 0.006 |
| Windows x64, private standard 2-core | 0.016 | 0.010 |

Sources: [pinned pre-change GitHub rate table](https://github.com/github/docs/blob/3d77b7e3ca336aa39ca90ab5e4a03434f17fe661/content/billing/reference/actions-runner-pricing.md),
[effective date](https://github.blog/changelog/2026-01-01-reduced-pricing-for-github-hosted-runners-usage/),
and [current prices](https://docs.github.com/en/billing/reference/actions-runner-pricing).
Historical jobs use the rate at their start date. Each executed job is rounded
up to whole minutes before pricing; no platform minute multiplier is applied
on top of dollar rates. GPU rates include GitHub's hosted platform fee.

BuildUSD includes 2,543 distinct executed jobs: 2,208 successful, 306 failed, and 29 cancelled.
Failed/cancelled jobs still consume compute. Skipped jobs cost zero.
Copied job records with identical run ID, name, start, and end are removed;
real reruns remain. Full-job times include setup, tests, and artifact handling,
not queue time. Timestamps approximate billing; no billable-usage export is available.
All cached runs have a job-fetch timestamp; this does not prove complete
historical retention. July 2025 and October 2026 are partial boundary months.

### Historical BuildUSD minutes and compute by component

| Component | Runner platform | Executed jobs | Rounded minutes | Public cost | Private proxy cost |
|---|---|---:|---:|---:|---:|
| validation | linux | 488 | 505 | $0.00 | $6.68 |
| linux | linux | 471 | 28,071 | $0.00 | $368.53 |
| windows | windows | 475 | 33,195 | $0.00 | $841.47 |
| macos | macos | 478 | 27,842 | $0.00 | $1,874.06 |
| wasm32 | linux | 291 | 14,803 | $0.00 | $177.64 |
| wasm64 | linux | 296 | 15,241 | $0.00 | $182.89 |
| linux_gpu | linux_gpu | 44 | 366 | $19.03 | $19.03 |

## Forecast: keep the existing BuildUSD workload mix

Use the existing low/mid/high trigger forecasts, including reruns (25%/50%/100%
of the observed release-level slope). Hold October 2026 prices constant for
all five years, with no inflation or extra growth factor. These are scenario
totals in current-price dollars, not confidence bounds.

Calibrate cost per trigger on **July-September 2026**, the latest three complete
calendar months. This includes both Wasm targets, Windows compiler caching,
GPU tests on eligible pushes, validation, failures, and selective reruns.
The baseline contains **193 inferred repo-wide events including reruns**
and **982 executed BuildUSD jobs**. The denominator is repo-wide
because that is the scope of the existing trigger forecast; packaging-only
events contribute zero BuildUSD cost. This preserves the observed participation
rate instead of charging every event for a full successful matrix.

Formula: forecast events * baseline component billed minutes / baseline events
* current component price. Failure and rerun overhead is already in this ratio;
do not multiply by a second failure/rerun allowance. Push/PR proportions and
cache-hit mix are assumed stable. 'Unchanged' means this observed workload mix
continues; it is not a replay benchmark of every old job on the latest YAML.

| Component | Baseline jobs | Rounded minutes per forecast event | Private proxy $/event |
|---|---:|---:|---:|
| validation | 165 | 0.891 | 0.0107 |
| linux | 160 | 53.212 | 0.6385 |
| windows | 161 | 27.363 | 0.6020 |
| macos | 160 | 48.440 | 3.0033 |
| wasm32 | 160 | 47.109 | 0.5653 |
| wasm64 | 160 | 48.197 | 0.5784 |
| linux_gpu | 16 | 0.648 | 0.0337 |

### One- and five-year cumulative costs

Year 1 means the first four projected three-month releases; five years means
all twenty. Each standardized month is 30.4375 days. Storage is included.

| Horizon | Scenario | Trigger events | Public | Private, resource-matched proxy | Private, same labels/times |
|---|---|---:|---:|---:|---:|
| 1 year(s) | low | 805.9 | $27.15 | $4,936.38 | $3,949.31 |
| 1 year(s) | mid | 893.7 | $30.10 | $5,462.85 | $4,368.24 |
| 1 year(s) | high | 1,069.3 | $36.02 | $6,515.86 | $5,206.18 |
| 5 year(s) | low | 5,434.7 | $183.06 | $33,286.32 | $26,629.85 |
| 5 year(s) | mid | 7,279.0 | $245.18 | $44,540.63 | $35,625.25 |
| 5 year(s) | high | 10,967.7 | $369.43 | $67,049.82 | $53,616.49 |

### Private proxy: platform cost breakdown (mid case)

| Component | Year 1 compute | Five-year compute | Year 1 artifact storage | Five-year artifact storage |
|---|---:|---:|---:|---:|
| validation | $9.56 | $77.84 | $0.00 | $0.00 |
| linux | $570.67 | $4,648.00 | $341.83 | $2,810.42 |
| windows | $537.99 | $4,381.81 | $34.63 | $284.74 |
| macos | $2,684.05 | $21,861.06 | $39.16 | $321.94 |
| wasm32 | $505.21 | $4,114.86 | $96.38 | $792.43 |
| wasm64 | $516.88 | $4,209.90 | $96.38 | $792.43 |
| linux_gpu | $30.10 | $245.15 | $0.00 | $0.03 |

### Baseline sensitivity

The windows below use the same event definition and current prices. They
illustrate runtime/mix sensitivity separately from forecast event-growth scenarios.

| Calibration window | Events | Private proxy compute $/event |
|---|---:|---:|
| 2026-07-01 to 2026-09-30 | 193 | 5.4319 |
| 2026-08-01 to 2026-09-30 | 141 | 5.1253 |
| 2026-09-01 to 2026-09-30 | 91 | 5.2949 |

## Artifact storage and exclusions

Artifacts stay in GitHub, not AWS/EBS. Use $0.25 per GiB-month and 90-day
retention; private storage is gross, with no plan allowance credited.
Public standard-job artifact storage is modeled as free under GitHub's public
Actions policy; conservatively price retained GPU failure artifacts at the
storage rate (less than one cent historically). Verify account billing if
reproducing an invoice. Sources: [Actions billing and storage](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
and [public Actions policy](https://docs.github.com/en/actions/concepts/billing-and-usage).

Historical artifact inputs: 799 retained metadata records,
261 imputed uploads with successful upload steps, and
724 older imputed uploads with only job-success evidence.
Known sizes and expiry timestamps are used directly. Missing uploads are
imputed from retained mean size for that build target and push/PR event type,
at job completion with 90-day retention. Wasm32 and Wasm64 are included.
Deleted artifacts and historical package/build changes make this approximate.
GPU failure artifacts without retained metadata cannot be reconstructed.
Historical charges accrue only within the observed window; post-window retention
and unknown pre-window artifacts are not charged into that historical total.

Forecast storage integrates a 90-day rolling window over the projected releases,
with an opening stock based on the recent baseline upload rate. This represents
taking over an existing workflow, not starting with empty storage. Each release's
uploads are spread uniformly; late-horizon uploads accrue only until the horizon.
The artifact CSV exposes observed versus imputed records for review.

No separate AWS egress charge is added to GitHub-hosted artifact transfers.
Cache overage, custom-image storage, subscriptions, taxes, labor, and unrelated
services are excluded. Assume the default 10 GiB repository cache cap rather than
an opt-in expanded paid cache. Hourly cache history and account settings are unknown.

## Other cached workflows: historical compute only

These are excluded from the BuildUSD forecasts above. They are material if
'footing the bill' means every workflow in the repository, particularly PyPiPackaging.

| Workflow | Linux minutes | Linux ARM minutes | Linux GPU minutes | Windows minutes | macOS minutes | Public compute | Private proxy compute |
|---|---:|---:|---:|---:|---:|---:|---:|
| BuildUSD | 58620 | 0 | 366 | 33195 | 27842 | $19.03 | $3,470.29 |
| Copilot code review | 4 | 0 | 0 | 0 | 0 | $0.00 | $0.05 |
| Graph Update: pip in /build_scripts/pypi/package_files #1373106687 | 1 | 0 | 0 | 0 | 0 | $0.00 | $0.01 |
| Graph Update: pip in /build_scripts/pypi/package_files #1540725403 | 2 | 0 | 0 | 0 | 0 | $0.00 | $0.02 |
| PyPiPackaging | 13814 | 759 | 0 | 19732 | 20855 | $0.00 | $2,076.87 |

This supplemental table is raw runner-rate repricing; it does not estimate
other-workflow artifact storage, Copilot AI charges, or account-specific exemptions.

## Data and reproduction

Inputs: local Dolt cache from the GitHub Actions REST API, current retained
artifact metadata fetched on 2026-10-02, and the committed trigger forecast.
No private billing access was used. The GPU label matches GitHub's published
4-core T4/28-GB-RAM/16-GB-VRAM larger-runner specification; pricing assumes that
published SKU rather than a special Pixar contract.

- [Job-level cost audit](openusd_workflow_costs_jobs.csv)
- [Monthly costs by workflow, component, and platform](openusd_workflow_costs_monthly.csv)
- [Forecast component costs and minutes](openusd_workflow_costs_forecast.csv)
- [Baseline coefficients](openusd_workflow_costs_baseline.csv)
- [Artifact upload audit](openusd_workflow_costs_artifact_uploads.csv)
- [Summary JSON](openusd_workflow_costs_summary.json)
- [Trigger forecast methodology](openusd_usage_forecast.md)

```sh
uv run python inspect_workflow_billing.py --fetch-artifacts
uv run python cache_runner_price_history.py
uv run python analyze_workflow_costs.py
```

Raw snapshots are cached locally in `.cache/workflow_billing/`. Analysis does
not modify the Dolt data, existing forecasts, or custom-matrix cost reports.
