#!/usr/bin/env python
"""Estimate retained GitHub workflow costs and forecast an unchanged BuildUSD mix.

Read locally cached public API inputs. No billing-account access is attempted.
"""

import argparse
import calendar
import csv
import datetime as dt
import json
import math
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from openusd_usage_dolt import infer_rerun_events, infer_source_events

ROOT = Path(__file__).resolve().parent
BASELINE_START = dt.datetime(2026, 7, 1)
BASELINE_END = dt.datetime(2026, 10, 1)
DAYS_PER_MONTH = 30.4375
RETENTION_DAYS = 90
STORAGE_RATE = 0.25
RATE_CHANGE = "2026-01-01"
# GitHub's published January 2026 rates, with its cloud platform fee included.
RATES = {
    "linux": (0.016, 0.012, 0.008, 0.006),
    "linux_arm": (0.010, 0.008, 0.005, 0.005),
    "windows": (0.032, 0.022, 0.016, 0.010),
    "macos": (0.080, 0.062, 0.080, 0.062),
    "linux_gpu": (0.070, 0.052, 0.070, 0.052),
}
FAMILIES = {
    "Validation": "validation",
    "Linux": "linux",
    "Windows": "windows",
    "macOS": "macos",
    "Wasm": "wasm32",
    "Wasm (Wasm)": "wasm32",
    "Wasm64": "wasm64",
    "Wasm (Wasm64)": "wasm64",
    "GPUTests": "linux_gpu",
}
ARTIFACT_FAMILIES = {
    "usd-linux": "linux",
    "usd-win64": "windows",
    "usd-macOS": "macos",
    "usd-Wasm": "wasm32",
    "usd-Wasm64": "wasm64",
    "FailedTestOutput": "linux_gpu",
}
SCENARIOS = ("low", "mid", "high")


def timestamp(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def platform_for(labels):
    if any("t4gpu" in label for label in labels):
        return "linux_gpu"
    if any(label.startswith("windows-") for label in labels):
        return "windows"
    if any(label.startswith("macos-") for label in labels):
        return "macos"
    if any(label.startswith("ubuntu-") for label in labels):
        return (
            "linux_arm" if any(label.endswith("-arm") for label in labels) else "linux"
        )
    raise ValueError(f"Unknown executed runner labels: {labels}")


def rate(platform, day, standard=False):
    return RATES[platform][(2 if standard else 0) + (day >= RATE_CHANGE)]


def executions(jobs):
    """Exclude unexecuted jobs and copied attempts, not failed execution time."""
    audit = []
    seen = set()
    for job in sorted(jobs, key=lambda r: (r["started_at"], int(r["job_id"]))):
        raw = json.loads(job["raw_json"])
        labels = json.loads(job["labels_json"])
        start, end = job["started_at"], job["completed_at"]
        seconds = (
            (timestamp(end) - timestamp(start)).total_seconds() if start and end else 0
        )
        identity = (job["run_id"], job["name"], start, end)
        reason = "included"
        if job["conclusion"] == "skipped":
            reason = "skipped"
        elif job["status"] != "completed":
            reason = "incomplete"
        elif not start or not end or seconds <= 0:
            reason = "no_positive_execution"
        elif not labels or (not raw.get("runner_name") and not job.get("runner_name")):
            reason = "no_runner_assigned"
        elif identity in seen:
            reason = "copied_execution"
        if reason == "included":
            seen.add(identity)
        platform = platform_for(labels) if reason == "included" else ""
        minutes = math.ceil(seconds / 60) if reason == "included" else 0
        audit.append(
            {
                "job_id": job["job_id"],
                "run_id": job["run_id"],
                "workflow": job["workflow_name"],
                "job_name": job["name"],
                "component": FAMILIES.get(job["name"], job["name"]),
                "event": job["event"],
                "platform": platform,
                "started_at": start,
                "completed_at": end,
                "attempt": raw.get("run_attempt", 1),
                "conclusion": job["conclusion"],
                "filter_result": reason,
                "elapsed_minutes": max(0, seconds / 60),
                "billed_minutes": minutes,
                "public_compute_usd": (
                    minutes * rate(platform, start[:10])
                    if platform == "linux_gpu"
                    else 0
                ),
                "private_equivalent_compute_usd": (
                    minutes * rate(platform, start[:10]) if platform else 0
                ),
                "private_same_label_compute_usd": (
                    minutes * rate(platform, start[:10], True) if platform else 0
                ),
            }
        )
    return audit


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def event_count(runs, jobs, start, end):
    events = infer_source_events(runs) + infer_rerun_events(runs, jobs)[0]
    return sum(start <= row["started_at"] < end for row in events)


def artifact_uploads(artifacts, rows, jobs, runs, cutoff):
    """Use known sizes for matching uploads; impute expired builds by event/family."""
    runs_by_id = {str(r["run_id"]): r for r in runs}
    raw_by_id = {str(j["job_id"]): json.loads(j["raw_json"]) for j in jobs}
    observed = []
    samples = defaultdict(list)
    by_run = defaultdict(list)
    for artifact in artifacts:
        run_id = str(artifact.get("workflow_run", {}).get("id", ""))
        run = runs_by_id.get(run_id)
        if not run or run["workflow_name"] != "BuildUSD":
            continue
        if artifact["name"] not in ARTIFACT_FAMILIES:
            raise ValueError(f"Unknown BuildUSD artifact: {artifact['name']}")
        start = timestamp(artifact["created_at"])
        if start >= cutoff:
            continue
        family = ARTIFACT_FAMILIES[artifact["name"]]
        size = artifact["size_in_bytes"] / 1024**3
        samples[(family, run["event"])].append(size)
        record = {
            "run_id": run_id,
            "component": family,
            "event": run["event"],
            "created_at": start,
            "gib": size,
            "expires_at": timestamp(artifact["expires_at"]),
            "source": "retained_metadata",
        }
        observed.append(record)
        by_run[(run_id, family)].append(record)
    output = list(observed)
    for row in rows:
        family = row["component"]
        if (
            row["workflow"] != "BuildUSD"
            or family not in set(ARTIFACT_FAMILIES.values())
            or family == "linux_gpu"
        ):
            continue
        start, end = timestamp(row["started_at"]), timestamp(row["completed_at"])
        if any(
            start <= a["created_at"] <= end + dt.timedelta(minutes=1)
            for a in by_run[(row["run_id"], family)]
        ):
            continue
        steps = raw_by_id[row["job_id"]].get("steps", [])
        uploads = [
            s
            for s in steps
            if "upload" in s["name"].lower() and "artifact" in s["name"].lower()
        ]
        if steps:
            uploaded = any(s.get("conclusion") == "success" for s in uploads)
        else:
            uploaded = row["conclusion"] == "success"
        if not uploaded:
            continue
        sizes = samples[(family, row["event"])]
        if not sizes:
            sizes = [
                v for (f, _), values in samples.items() if f == family for v in values
            ]
        if not sizes:
            raise ValueError(f"No artifact size proxy for {family}")
        output.append(
            {
                "run_id": row["run_id"],
                "component": family,
                "event": row["event"],
                "created_at": end,
                "gib": statistics.mean(sizes),
                "expires_at": end + dt.timedelta(days=RETENTION_DAYS),
                "source": "imputed_upload_step" if steps else "imputed_job_success",
            }
        )
    return output


def storage_cost(uploads, start, end):
    """Accrue GiB-hours within the reporting window, using calendar-month hours."""
    total = 0.0
    for artifact in uploads:
        left = max(start, artifact["created_at"])
        right = min(end, artifact["expires_at"])
        while left < right:
            following = (left.replace(day=1) + dt.timedelta(days=32)).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            stop = min(right, following)
            month_hours = calendar.monthrange(left.year, left.month)[1] * 24
            total += (
                artifact["gib"]
                * (stop - left).total_seconds()
                / 3600
                / month_hours
                * STORAGE_RATE
            )
            left = stop
    return total


def retained_gib_days(upload_start, upload_end, horizon_end):
    """Integral of remaining retention inside a horizon for constant daily uploads."""
    points = sorted(
        {
            upload_start,
            upload_end,
            *[
                p
                for p in (0, RETENTION_DAYS, horizon_end, horizon_end - RETENTION_DAYS)
                if upload_start < p < upload_end
            ],
        }
    )

    def life(day):
        return max(0, min(day + RETENTION_DAYS, horizon_end) - max(day, 0))

    return sum((b - a) * (life(a) + life(b)) / 2 for a, b in zip(points, points[1:]))


def forecast_storage(releases, scenario, gib_per_event, baseline_daily_events, years):
    horizon = years * 12 * DAYS_PER_MONTH
    gib_days = (
        baseline_daily_events
        * gib_per_event
        * retained_gib_days(-RETENTION_DAYS, 0, horizon)
    )
    release_days = 3 * DAYS_PER_MONTH
    for index, row in enumerate(releases[: years * 4]):
        start = index * release_days
        events = float(row[f"{scenario}_events_including_reruns_per_release"])
        gib_days += (
            events
            / release_days
            * gib_per_event
            * retained_gib_days(start, start + release_days, horizon)
        )
    return gib_days / DAYS_PER_MONTH * STORAGE_RATE


def analyze(cache, output):
    jobs, runs, artifacts, metadata = [
        json.loads((cache / f"{name}.json").read_text())
        for name in ("jobs", "runs", "artifacts", "metadata")
    ]
    cutoff = timestamp(
        next(r["updated_at"] for r in metadata if r["meta_key"] == "actions_job_count")
    )
    audit = executions(jobs)
    included = [r for r in audit if r["filter_result"] == "included"]
    core = [r for r in included if r["workflow"] == "BuildUSD"]
    start = min(timestamp(r["started_at"]) for r in core)
    uploads = artifact_uploads(artifacts, core, jobs, runs, cutoff)
    baseline_n = event_count(runs, jobs, BASELINE_START, BASELINE_END)
    baseline = [
        r for r in core if BASELINE_START <= timestamp(r["started_at"]) < BASELINE_END
    ]
    baseline_artifacts = [
        a for a in uploads if BASELINE_START <= a["created_at"] < BASELINE_END
    ]
    with (ROOT / "openusd_triggering_events_forecast.csv").open() as stream:
        releases = list(csv.DictReader(stream))
    monthly = defaultdict(
        lambda: {
            "jobs": 0,
            "elapsed_minutes": 0.0,
            "billed_minutes": 0,
            "public_compute_usd": 0.0,
            "private_equivalent_compute_usd": 0.0,
            "private_same_label_compute_usd": 0.0,
        }
    )
    for row in included:
        key = (
            row["started_at"][:7],
            row["workflow"],
            row["platform"],
            row["component"],
        )
        monthly[key]["jobs"] += 1
        for column in monthly[key]:
            if column != "jobs":
                monthly[key][column] += row[column]
    historical = [
        {
            "month": key[0],
            "workflow": key[1],
            "platform": key[2],
            "component": key[3],
            **value,
        }
        for key, value in sorted(monthly.items())
    ]
    coefficients = []
    for component in FAMILIES.values():
        if component in {r["component"] for r in coefficients}:
            continue
        selected = [r for r in baseline if r["component"] == component]
        if not selected:
            raise ValueError(f"Missing baseline: {component}")
        platform = selected[0]["platform"]
        minutes = sum(r["billed_minutes"] for r in selected) / baseline_n
        coefficients.append(
            {
                "component": component,
                "platform": platform,
                "baseline_jobs": len(selected),
                "baseline_trigger_events": baseline_n,
                "billed_minutes_per_event": minutes,
                "artifact_gib_per_event": (
                    sum(
                        a["gib"]
                        for a in baseline_artifacts
                        if a["component"] == component
                    )
                    / baseline_n
                ),
                "public_usd_per_event": (
                    minutes * rate(platform, RATE_CHANGE)
                    if platform == "linux_gpu"
                    else 0
                ),
                "private_equivalent_usd_per_event": (
                    minutes * rate(platform, RATE_CHANGE)
                ),
                "private_same_label_usd_per_event": (
                    minutes * rate(platform, RATE_CHANGE, True)
                ),
            }
        )
    forecast = []
    for years in (1, 2, 3, 4, 5):
        for scenario in SCENARIOS:
            count = sum(
                float(r[f"{scenario}_events_including_reruns_per_release"])
                for r in releases[: years * 4]
            )
            for row in coefficients:
                storage = forecast_storage(
                    releases,
                    scenario,
                    row["artifact_gib_per_event"],
                    baseline_n / (BASELINE_END - BASELINE_START).days,
                    years,
                )
                forecast.append(
                    {
                        "years": years,
                        "scenario": scenario,
                        "component": row["component"],
                        "platform": row["platform"],
                        "events_including_reruns": count,
                        "billed_minutes": count * row["billed_minutes_per_event"],
                        "public_compute_usd": count * row["public_usd_per_event"],
                        "private_equivalent_compute_usd": (
                            count * row["private_equivalent_usd_per_event"]
                        ),
                        "private_same_label_compute_usd": (
                            count * row["private_same_label_usd_per_event"]
                        ),
                        "public_artifact_storage_usd": (
                            storage if row["component"] == "linux_gpu" else 0
                        ),
                        "private_artifact_storage_usd": storage,
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("jobs", audit),
        ("monthly", historical),
        ("baseline", coefficients),
        ("forecast", forecast),
        ("artifact_uploads", uploads),
    ):
        write_csv(output / f"openusd_workflow_costs_{name}.csv", rows)
    summary = {
        "start": str(start),
        "end": str(cutoff),
        "filters": dict(Counter(r["filter_result"] for r in audit)),
        "buildusd_jobs": len(core),
        "buildusd_conclusions": dict(Counter(r["conclusion"] for r in core)),
        "baseline_events": baseline_n,
        "baseline_jobs": len(baseline),
        "historical_artifact_storage_gross_usd": storage_cost(uploads, start, cutoff),
        "historical_public_gpu_artifact_storage_gross_usd": storage_cost(
            [a for a in uploads if a["component"] == "linux_gpu"], start, cutoff
        ),
        "artifact_sources": dict(Counter(a["source"] for a in uploads)),
        "missing_job_fetch": sum(not r["jobs_fetched_at"] for r in runs),
        "historical_compute": {
            scope: {
                field: sum(r[field] for r in records)
                for field in (
                    "public_compute_usd",
                    "private_equivalent_compute_usd",
                    "private_same_label_compute_usd",
                )
            }
            for scope, records in (
                ("BuildUSD", core),
                ("all_cached_workflows", included),
            )
        },
    }
    (output / "openusd_workflow_costs_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    report(output, summary, core, included, coefficients, forecast, uploads, runs, jobs)
    print(json.dumps(summary, indent=2, sort_keys=True))
    for years in (1, 5):
        for scenario in SCENARIOS:
            rows = [
                r for r in forecast if r["years"] == years and r["scenario"] == scenario
            ]
            print(
                "FORECAST",
                years,
                scenario,
                {
                    k: round(sum(r[k] for r in rows), 2)
                    for k in (
                        "public_compute_usd",
                        "private_equivalent_compute_usd",
                        "private_same_label_compute_usd",
                        "private_artifact_storage_usd",
                    )
                },
            )


def report(
    output, summary, core, included, coefficients, forecast, uploads, runs, jobs
):
    def money(value):
        return f"${value:,.2f}"

    totals = summary["historical_compute"]["BuildUSD"]
    storage = summary["historical_artifact_storage_gross_usd"]
    public_storage = summary["historical_public_gpu_artifact_storage_gross_usd"]
    lines = [
        "# Existing OpenUSD workflow: historical costs and projections",
        "",
        "Analysis date: 2026-10-02. Currency: USD.",
        "",
        "## Headline: BuildUSD, not the custom 16-configuration matrix",
        "",
        (
            "Workflow reference: [BuildUSD at the latest cached build"
            " commit](https://github.com/PixarAnimationStudios/OpenUSD/blob/240e3ddef8d79a5800490c85ca4a7741397f8516/.github/workflows/buildusd.yml)."
        ),
        "",
        (
            f"Retained execution window: **{summary['start']} through {summary['end']}"
            " UTC**."
        ),
        "This is a public-data list-price estimate, not Pixar's invoice. Discounts,",
        "credits, sponsorship, enterprise seats, and account-level storage settings",
        (
            "are not visible. Expired/deleted runs outside the cache cannot be"
            " reconstructed."
        ),
        "",
        (
            "| Historical BuildUSD cost | Public repository | Private, resource-matched"
            " proxy | Private, same labels and recorded times |"
        ),
        "|---|---:|---:|---:|",
        (
            f"| Runner compute | {money(totals['public_compute_usd'])} |"
            f" {money(totals['private_equivalent_compute_usd'])} |"
            f" {money(totals['private_same_label_compute_usd'])} |"
        ),
        (
            f"| Estimated artifact storage | {money(public_storage)} | {money(storage)}"
            f" | {money(storage)} |"
        ),
        (
            "| Total of modeled items |"
            f" **{money(totals['public_compute_usd'] + public_storage)}** |"
            f" **{money(totals['private_equivalent_compute_usd'] + storage)}** |"
            f" **{money(totals['private_same_label_compute_usd'] + storage)}** |"
        ),
        "",
        "The public compute charge is entirely the Linux GPU-test runner. Standard",
        "Linux, Windows, macOS, validation, and Wasm jobs are free in the public case.",
        "Private cases apply no included minutes or artifact allowance, as requested.",
        "They are marginal costs with the account allowance exhausted, not a statement",
        "that GitHub private plans have no allowance.",
        "",
        "## What 'equivalent paid runners' means",
        "",
        (
            "[Standard runner"
            " specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)"
        ),
        "give public Linux/Windows labels 4 vCPUs and 16 GB RAM, versus 2 vCPUs and",
        "8 GB in private repositories. macOS retains the same standard shape.",
        "The main proxy prices recorded times at paid 4-core Linux/Windows rates.",
        "This would require runner routing changes, not an untouched YAML file.",
        "",
        (
            "Important: the [larger-runner"
            " reference](https://docs.github.com/en/actions/reference/runners/larger-runners)"
        ),
        "limits paid 4-core Windows runners to Windows Server 2025 or Windows 11,",
        "not the workflow's Windows Server 2022. Thus the 4-core Windows amount is",
        "a CPU/RAM-matched price proxy, not an exact purchasable 2022 equivalent.",
        "The same-label column leaves runner labels unchanged, but reuses the public",
        (
            "durations on smaller Linux/Windows VMs. It is a time-held-constant"
            " comparison,"
        ),
        "not a validated budget: runtimes, failures, and cache behavior can change.",
        "No unsupported 2x runtime assumption is silently applied.",
        "",
        "## Rates and historical accounting",
        "",
        "| Runner | Before 2026, $/minute | Since 2026-01-01, $/minute |",
        "|---|---:|---:|",
        "| Linux x64, paid 4-core proxy | 0.016 | 0.012 |",
        "| Windows x64, paid 4-core proxy | 0.032 | 0.022 |",
        "| macOS standard | 0.080 | 0.062 |",
        "| Linux T4 GPU, 4-core | 0.070 | 0.052 |",
        "| Linux x64, private standard 2-core | 0.008 | 0.006 |",
        "| Windows x64, private standard 2-core | 0.016 | 0.010 |",
        "",
        (
            "Sources: [pinned pre-change GitHub rate"
            " table](https://github.com/github/docs/blob/3d77b7e3ca336aa39ca90ab5e4a03434f17fe661/content/billing/reference/actions-runner-pricing.md),"
        ),
        (
            "[effective"
            " date](https://github.blog/changelog/2026-01-01-reduced-pricing-for-github-hosted-runners-usage/),"
        ),
        (
            "and [current"
            " prices](https://docs.github.com/en/billing/reference/actions-runner-pricing)."
        ),
        (
            "Historical jobs use the rate at their start date. Each executed job is"
            " rounded"
        ),
        "up to whole minutes before pricing; no platform minute multiplier is applied",
        "on top of dollar rates. GPU rates include GitHub's hosted platform fee.",
        "",
        (
            f"BuildUSD includes {len(core):,} distinct executed jobs: "
            f"{summary['buildusd_conclusions']['success']:,} successful, "
            f"{summary['buildusd_conclusions']['failure']:,} failed, and "
            f"{summary['buildusd_conclusions']['cancelled']:,} cancelled."
        ),
        "Failed/cancelled jobs still consume compute. Skipped jobs cost zero.",
        "Copied job records with identical run ID, name, start, and end are removed;",
        (
            "real reruns remain. Full-job times include setup, tests, and artifact"
            " handling,"
        ),
        (
            "not queue time. Timestamps approximate billing; no billable-usage export"
            " is available."
        ),
        "All cached runs have a job-fetch timestamp; this does not prove complete",
        "historical retention. July 2025 and October 2026 are partial boundary months.",
        "",
        "### Historical BuildUSD minutes and compute by component",
        "",
        (
            "| Component | Runner platform | Executed jobs | Rounded minutes | Public"
            " cost | Private proxy cost |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]
    for component in coefficients:
        rows = [r for r in core if r["component"] == component["component"]]
        lines.append(
            f"| {component['component']} | {component['platform']} | {len(rows)} |"
            f" {sum(r['billed_minutes'] for r in rows):,} |"
            f" {money(sum(r['public_compute_usd'] for r in rows))} |"
            f" {money(sum(r['private_equivalent_compute_usd'] for r in rows))} |"
        )
    lines += [
        "",
        "## Forecast: keep the existing BuildUSD workload mix",
        "",
        (
            "Use the existing low/mid/high trigger forecasts, including reruns"
            " (25%/50%/100%"
        ),
        "of the observed release-level slope). Hold October 2026 prices constant for",
        "all five years, with no inflation or extra growth factor. These are scenario",
        "totals in current-price dollars, not confidence bounds.",
        "",
        (
            "Calibrate cost per trigger on **July-September 2026**, the latest three"
            " complete"
        ),
        "calendar months. This includes both Wasm targets, Windows compiler caching,",
        "GPU tests on eligible pushes, validation, failures, and selective reruns.",
        (
            f"The baseline contains **{summary['baseline_events']} inferred repo-wide"
            " events including reruns**"
        ),
        (
            f"and **{summary['baseline_jobs']} executed BuildUSD jobs**. The"
            " denominator is repo-wide"
        ),
        "because that is the scope of the existing trigger forecast; packaging-only",
        (
            "events contribute zero BuildUSD cost. This preserves the observed"
            " participation"
        ),
        "rate instead of charging every event for a full successful matrix.",
        "",
        (
            "Formula: forecast events * baseline component billed minutes / baseline"
            " events"
        ),
        (
            "* current component price. Failure and rerun overhead is already in this"
            " ratio;"
        ),
        "do not multiply by a second failure/rerun allowance. Push/PR proportions and",
        (
            "cache-hit mix are assumed stable. 'Unchanged' means this observed"
            " workload mix"
        ),
        "continues; it is not a replay benchmark of every old job on the latest YAML.",
        "",
        (
            "| Component | Baseline jobs | Rounded minutes per forecast event | Private"
            " proxy $/event |"
        ),
        "|---|---:|---:|---:|",
    ]
    for row in coefficients:
        lines.append(
            f"| {row['component']} | {row['baseline_jobs']} |"
            f" {row['billed_minutes_per_event']:.3f} |"
            f" {row['private_equivalent_usd_per_event']:.4f} |"
        )
    lines += [
        "",
        "### One- and five-year cumulative costs",
        "",
        "Year 1 means the first four projected three-month releases; five years means",
        "all twenty. Each standardized month is 30.4375 days. Storage is included.",
        "",
        (
            "| Horizon | Scenario | Trigger events | Public | Private, resource-matched"
            " proxy | Private, same labels/times |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]
    for years in (1, 5):
        for scenario in SCENARIOS:
            rows = [
                r for r in forecast if r["years"] == years and r["scenario"] == scenario
            ]
            public = sum(
                r["public_compute_usd"] + r["public_artifact_storage_usd"] for r in rows
            )
            equivalent = sum(
                r["private_equivalent_compute_usd"] + r["private_artifact_storage_usd"]
                for r in rows
            )
            literal = sum(
                r["private_same_label_compute_usd"] + r["private_artifact_storage_usd"]
                for r in rows
            )
            lines.append(
                f"| {years} year(s) | {scenario} |"
                f" {rows[0]['events_including_reruns']:,.1f} | {money(public)} |"
                f" {money(equivalent)} | {money(literal)} |"
            )
    lines += [
        "",
        "### Private proxy: platform cost breakdown (mid case)",
        "",
        (
            "| Component | Year 1 compute | Five-year compute | Year 1 artifact storage"
            " | Five-year artifact storage |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for row in coefficients:
        one, five = [
            next(
                r
                for r in forecast
                if r["years"] == years
                and r["scenario"] == "mid"
                and r["component"] == row["component"]
            )
            for years in (1, 5)
        ]
        lines.append(
            f"| {row['component']} | {money(one['private_equivalent_compute_usd'])} |"
            f" {money(five['private_equivalent_compute_usd'])} |"
            f" {money(one['private_artifact_storage_usd'])} |"
            f" {money(five['private_artifact_storage_usd'])} |"
        )
    lines += [
        "",
        "### Baseline sensitivity",
        "",
        "The windows below use the same event definition and current prices. They",
        (
            "illustrate runtime/mix sensitivity separately from forecast event-growth"
            " scenarios."
        ),
        "",
        "| Calibration window | Events | Private proxy compute $/event |",
        "|---|---:|---:|",
    ]
    for start in (BASELINE_START, dt.datetime(2026, 8, 1), dt.datetime(2026, 9, 1)):
        count = event_count(runs, jobs, start, BASELINE_END)
        cost = sum(
            r["billed_minutes"] * rate(r["platform"], RATE_CHANGE)
            for r in core
            if start <= timestamp(r["started_at"]) < BASELINE_END
        )
        lines.append(f"| {start.date()} to 2026-09-30 | {count} | {cost / count:.4f} |")
    lines += [
        "",
        "## Artifact storage and exclusions",
        "",
        "Artifacts stay in GitHub, not AWS/EBS. Use $0.25 per GiB-month and 90-day",
        "retention; private storage is gross, with no plan allowance credited.",
        "Public standard-job artifact storage is modeled as free under GitHub's public",
        "Actions policy; conservatively price retained GPU failure artifacts at the",
        "storage rate (less than one cent historically). Verify account billing if",
        (
            "reproducing an invoice. Sources: [Actions billing and"
            " storage](https://docs.github.com/en/billing/concepts/product-billing/github-actions)"
        ),
        (
            "and [public Actions"
            " policy](https://docs.github.com/en/actions/concepts/billing-and-usage)."
        ),
        "",
        (
            "Historical artifact inputs:"
            f" {summary['artifact_sources']['retained_metadata']} retained metadata"
            " records,"
        ),
        (
            f"{summary['artifact_sources']['imputed_upload_step']} imputed uploads with"
            " successful upload steps, and"
        ),
        (
            f"{summary['artifact_sources']['imputed_job_success']} older imputed"
            " uploads with only job-success evidence."
        ),
        "Known sizes and expiry timestamps are used directly. Missing uploads are",
        "imputed from retained mean size for that build target and push/PR event type,",
        "at job completion with 90-day retention. Wasm32 and Wasm64 are included.",
        "Deleted artifacts and historical package/build changes make this approximate.",
        "GPU failure artifacts without retained metadata cannot be reconstructed.",
        (
            "Historical charges accrue only within the observed window; post-window"
            " retention"
        ),
        "and unknown pre-window artifacts are not charged into that historical total.",
        "",
        (
            "Forecast storage integrates a 90-day rolling window over the projected"
            " releases,"
        ),
        (
            "with an opening stock based on the recent baseline upload rate. This"
            " represents"
        ),
        (
            "taking over an existing workflow, not starting with empty storage. Each"
            " release's"
        ),
        (
            "uploads are spread uniformly; late-horizon uploads accrue only until the"
            " horizon."
        ),
        "The artifact CSV exposes observed versus imputed records for review.",
        "",
        "No separate AWS egress charge is added to GitHub-hosted artifact transfers.",
        (
            "Cache overage, custom-image storage, subscriptions, taxes, labor, and"
            " unrelated"
        ),
        (
            "services are excluded. Assume the default 10 GiB repository cache cap"
            " rather than"
        ),
        (
            "an opt-in expanded paid cache. Hourly cache history and account settings"
            " are unknown."
        ),
        "",
        "## Other cached workflows: historical compute only",
        "",
        "These are excluded from the BuildUSD forecasts above. They are material if",
        (
            "'footing the bill' means every workflow in the repository, particularly"
            " PyPiPackaging."
        ),
        "",
        (
            "| Workflow | Linux minutes | Linux ARM minutes | Linux GPU minutes |"
            " Windows minutes | macOS minutes | Public compute | Private proxy"
            " compute |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for workflow in sorted({r["workflow"] for r in included}):
        rows = [r for r in included if r["workflow"] == workflow]
        minutes = [
            str(sum(r["billed_minutes"] for r in rows if r["platform"] == platform))
            for platform in ("linux", "linux_arm", "linux_gpu", "windows", "macos")
        ]
        lines.append(
            f"| {workflow} | "
            + " | ".join(minutes)
            + f" | {money(sum(r['public_compute_usd'] for r in rows))} |"
            f" {money(sum(r['private_equivalent_compute_usd'] for r in rows))} |"
        )
    lines += [
        "",
        "This supplemental table is raw runner-rate repricing; it does not estimate",
        (
            "other-workflow artifact storage, Copilot AI charges, or account-specific"
            " exemptions."
        ),
        "",
        "## Data and reproduction",
        "",
        "Inputs: local Dolt cache from the GitHub Actions REST API, current retained",
        "artifact metadata fetched on 2026-10-02, and the committed trigger forecast.",
        "No private billing access was used. The GPU label matches GitHub's published",
        (
            "4-core T4/28-GB-RAM/16-GB-VRAM larger-runner specification; pricing"
            " assumes that"
        ),
        "published SKU rather than a special Pixar contract.",
        "",
        "- [Job-level cost audit](openusd_workflow_costs_jobs.csv)",
        (
            "- [Monthly costs by workflow, component, and"
            " platform](openusd_workflow_costs_monthly.csv)"
        ),
        "- [Forecast component costs and minutes](openusd_workflow_costs_forecast.csv)",
        "- [Baseline coefficients](openusd_workflow_costs_baseline.csv)",
        "- [Artifact upload audit](openusd_workflow_costs_artifact_uploads.csv)",
        "- [Summary JSON](openusd_workflow_costs_summary.json)",
        "- [Trigger forecast methodology](openusd_usage_forecast.md)",
        "",
        "```sh",
        "uv run python inspect_workflow_billing.py --fetch-artifacts",
        "uv run python cache_runner_price_history.py",
        "uv run python analyze_workflow_costs.py",
        "```",
        "",
        "Raw snapshots are cached locally in `.cache/workflow_billing/`. Analysis does",
        "not modify the Dolt data, existing forecasts, or custom-matrix cost reports.",
    ]
    (output / "openusd_workflow_costs.md").write_text("\n".join(lines) + "\n")


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / ".cache/workflow_billing"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    try:
        analyze(args.cache_dir, args.output_dir)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
