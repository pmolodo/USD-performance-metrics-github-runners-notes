#!/usr/bin/env python

"""Summarize successful OpenUSD build-job durations by platform and UTC month."""

import argparse
import calendar
import csv
import datetime as dt
import io
import json
import statistics
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path

PLATFORMS = ("linux", "windows", "macos", "wasm32", "wasm64")
TITLES = dict(zip(PLATFORMS, ("Linux x64", "Windows", "macOS", "Wasm32", "Wasm64")))
JOB_NAMES = {
    "Linux": "linux",
    "Windows": "windows",
    "macOS": "macos",
    "Wasm": "wasm32",
    "Wasm (Wasm)": "wasm32",
    "Wasm64": "wasm64",
    "Wasm (Wasm64)": "wasm64",
}
ROOT = Path(__file__).resolve().parent
METRICS = ("job_minutes", "build_step_minutes")
# Commit 528289b55f035f660f852c69fe930ebe50dac6fc introduces ccache-action,
# --compiler-cache, and the Ninja generator for Windows. Use its UTC commit date.
WINDOWS_CACHE_DATE = dt.date(2026, 5, 28)
WINDOWS_CACHE_COMMIT = "528289b55f035f660f852c69fe930ebe50dac6fc"
PANEL_PLATFORMS = ("windows", "wasm32", "linux", "wasm64", "macos")


def query(database, sql):
    result = subprocess.run(
        ["dolt", "sql", "-r", "csv", "-q", sql],
        cwd=database,
        check=True,
        capture_output=True,
        text=True,
    )
    return list(csv.DictReader(io.StringIO(result.stdout)))


def duration(start, end):
    if not start or not end:
        return None
    value = (
        dt.datetime.fromisoformat(end.replace("Z", "+00:00"))
        - dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
    ).total_seconds() / 60
    return value if value > 0 else None


def summarize(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "p10": "", "p50": "", "p90": "", "mean": ""}

    # Inclusive linear interpolation: rank (n - 1) * p; defined even for n=1.
    def percentile(p):
        position = (len(values) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {
        "n": len(values),
        "p10": percentile(0.1),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "mean": statistics.mean(values),
    }


def classify(rows):
    audit = []
    seen = set()
    for row in rows:
        if row["name"] not in JOB_NAMES:
            continue
        raw = json.loads(row["raw_json"])
        steps = raw.get("steps", [])
        build_steps = [s for s in steps if s["name"].lower() == "build usd"]
        minutes = duration(row["started_at"], row["completed_at"])
        reason = "included"
        if row["status"] != "completed" or row["conclusion"] != "success":
            reason = "not_successful"
        elif minutes is None:
            reason = "invalid_job_timestamps"
        elif steps and (
            not build_steps
            or any(s.get("conclusion") != "success" for s in build_steps)
        ):
            reason = "build_step_not_successful_or_missing"
        elif any(
            s.get("conclusion") in ("failure", "cancelled", "timed_out") for s in steps
        ):
            reason = "failed_step_inside_successful_job"
        # GitHub can return copied jobs from earlier attempts with new IDs.
        # Different start/end times remain distinct, including real reruns.
        identity = (row["run_id"], row["name"], row["started_at"], row["completed_at"])
        if reason == "included":
            if identity in seen:
                reason = "duplicate_execution"
            else:
                seen.add(identity)
        build_minutes = (
            duration(
                build_steps[0].get("started_at"), build_steps[0].get("completed_at")
            )
            if build_steps
            else None
        )
        audit.append(
            {
                "job_id": row["job_id"],
                "run_id": row["run_id"],
                "run_attempt": raw.get("run_attempt", ""),
                "platform": JOB_NAMES[row["name"]],
                "job_name": row["name"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "month": row["started_at"][:7],
                "conclusion": row["conclusion"],
                "filter_result": reason,
                "steps_available": bool(steps),
                "job_minutes": minutes if minutes is not None else "",
                "build_step_minutes": (
                    build_minutes if build_minutes is not None else ""
                ),
                "runner_labels": row["labels_json"],
                "url": (
                    raw.get("html_url")
                    or f"https://github.com/PixarAnimationStudios/OpenUSD/actions/runs/{row['run_id']}"
                ),
            }
        )
    return audit


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, metric, output, cache_month):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"svg.fonttype": "none", "font.size": 10})
    months = sorted({r["month"] for r in rows})
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, sharey=True)
    for axis, platform in zip(axes.flat, PANEL_PLATFORMS):
        selected = {
            r["month"]: r
            for r in rows
            if r["platform"] == platform and r["metric"] == metric
        }
        series = [selected.get(month, {}) for month in months]

        def values(key):
            return [r.get(key) if r.get("n", 0) else float("nan") for r in series]

        axis.fill_between(
            range(len(months)), values("p10"), values("p90"), color="#b5d5f0", alpha=0.6
        )
        for key, color, style in (
            ("p10", "#6699bb", ":"),
            ("p90", "#6699bb", ":"),
            ("p50", "#1263a0", "-"),
            ("mean", "#c55a11", "--"),
        ):
            axis.plot(
                range(len(months)),
                values(key),
                color=color,
                linestyle=style,
                marker="o" if key in ("p50", "mean") else None,
                markersize=3,
            )
        axis.set_title(TITLES[platform], loc="left", fontweight="bold")
        axis.set_ylabel("Minutes per successful build")
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(
            range(len(months)),
            [m + ("*" if m == cache_month else "") for m in months],
            rotation=55,
            ha="right",
        )
        axis.tick_params(labelbottom=True)
        if platform == "windows" and WINDOWS_CACHE_DATE.strftime("%Y-%m") in months:
            # Month ticks represent month starts; interpolate the event date
            # within May instead of implying that all May jobs used caching.
            index = months.index(WINDOWS_CACHE_DATE.strftime("%Y-%m"))
            position = (
                index
                + (WINDOWS_CACHE_DATE.day - 1)
                / calendar.monthrange(
                    WINDOWS_CACHE_DATE.year, WINDOWS_CACHE_DATE.month
                )[1]
            )
            marker = axis.axvline(
                position,
                color="#80649d",
                alpha=0.5,
                linewidth=1.2,
                linestyle="--",
                zorder=3,
            )
            marker.set_gid("windows-compiler-cache-enabled")
            axis.text(
                position + 0.12,
                0.86,
                f"ccache + Ninja enabled\n{WINDOWS_CACHE_DATE.isoformat()}",
                transform=axis.get_xaxis_transform(),
                va="top",
                fontsize=9,
                color="#65467f",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.85,
                    "edgecolor": "none",
                    "pad": 2,
                },
            )
        for i, row in enumerate(series):
            if row.get("n", 0):
                axis.text(
                    i,
                    0.97,
                    str(row["n"]),
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#555555",
                )
        if cache_month in months:
            index = months.index(cache_month)
            axis.axvspan(
                index - 0.45, index + 0.45, color="#fff2b2", alpha=0.5, zorder=0
            )
    maximum = max(
        max(r["p90"], r["mean"]) for r in rows if r["metric"] == metric and r["n"]
    )
    axes.flat[0].set_ylim(0, maximum * 1.18)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(
        handles=[
            Line2D([0], [0], color="#1263a0", marker="o", label="Median (p50)"),
            Line2D([0], [0], color="#c55a11", linestyle="--", label="Arithmetic mean"),
            Line2D(
                [0],
                [0],
                color="#6699bb",
                linestyle=":",
                label="p10 / p90; shaded interval",
            ),
        ],
        loc="upper left",
        frameon=False,
    )
    axes.flat[-1].text(
        0.03,
        0.58,
        "Top labels: sample count per month\n* Cache month is partial\nGaps mean no"
        " usable samples, not zero time\n\nSuccessful attempts only; copied executions"
        " deduplicated\nReal reruns with distinct execution times retained\nNo"
        " duration-based trimming of successful jobs\n\nJuly 2025 is partial; older"
        " cache coverage is incomplete.\nPercentiles describe observed jobs, not"
        " confidence bounds.",
        transform=axes.flat[-1].transAxes,
        va="top",
        linespacing=1.35,
        fontsize=9,
    )
    title = (
        "Full job elapsed time"
        if metric == "job_minutes"
        else "Build USD step elapsed time (available steps only)"
    )
    fig.suptitle(f"OpenUSD successful builds by UTC month\n{title}", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, facecolor="white", transparent=False)
    if metric == "job_minutes":
        fig.savefig(output.with_suffix(".png"), facecolor="white", dpi=100)
    plt.close(fig)


def analyze(database, output):
    rows = query(
        database,
        "SELECT j.* FROM workflow_jobs j JOIN workflow_runs r ON j.run_id=r.run_id "
        "WHERE r.workflow_name='BuildUSD' ORDER BY j.started_at, j.job_id",
    )
    metadata = {
        r["meta_key"]: r for r in query(database, "SELECT * FROM sync_metadata")
    }
    fetched = metadata["actions_job_count"]["updated_at"]
    audit = classify(rows)
    included = [r for r in audit if r["filter_result"] == "included"]
    if not included:
        raise ValueError("No valid successful build jobs")
    monthly = []
    overall = []
    months = sorted({r["month"] for r in included})
    for platform in PLATFORMS:
        for metric in METRICS:
            relevant = [
                r for r in included if r["platform"] == platform and r[metric] != ""
            ]
            overall.append(
                {
                    "platform": platform,
                    "metric": metric,
                    **summarize([r[metric] for r in relevant]),
                }
            )
            for month in months:
                selected = [r for r in relevant if r["month"] == month]
                monthly.append(
                    {
                        "month": month,
                        "platform": platform,
                        "metric": metric,
                        **summarize([r[metric] for r in selected]),
                        "partial_boundary_month": month in (months[0], fetched[:7]),
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("jobs", audit), ("monthly", monthly), ("summary", overall)):
        write_csv(output / f"openusd_build_timings_{name}.csv", data)
    for metric, name in zip(METRICS, ("monthly", "build_step_monthly")):
        plot(monthly, metric, output / f"openusd_build_timings_{name}.svg", fetched[:7])
    counts = Counter(r["filter_result"] for r in audit)
    lines = [
        "# OpenUSD successful build durations",
        "",
        f"Actions cache collected: {fetched} UTC.",
        "",
        "Platforms: Linux x64, Windows, macOS, Wasm32, Wasm64. Wasm runs on Linux,",
        "but is kept separate here as a target platform. GPU-only tests and validation",
        "jobs are excluded. Native coverage begins July 31, 2025; wasm begins February",
        (
            "2026. October 2026 is partial. This is retained cache coverage, not proof"
            " that"
        ),
        "earlier builds did not run. No new GitHub collection was performed.",
        "",
        "## Successful full-job duration, minutes",
        "",
        "| Platform | Samples | p10 | Median | p90 | Mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        if row["metric"] == "job_minutes":
            lines.append(
                f"| {TITLES[row['platform']]} | {row['n']} | {row['p10']:.2f} |"
                f" {row['p50']:.2f} | {row['p90']:.2f} | {row['mean']:.2f} |"
            )
    lines += [
        "",
        "![Monthly successful job timings](openusd_build_timings_monthly.svg)",
        "",
        (
            f"Windows' vertical marker is {WINDOWS_CACHE_DATE.isoformat()}, the UTC"
            " commit date"
        ),
        "that introduced persistent ccache and switched to Ninja:",
        (
            "[workflow"
            f" change](https://github.com/PixarAnimationStudios/OpenUSD/commit/{WINDOWS_CACHE_COMMIT})."
        ),
        "Monthly points mix jobs before and after a change; this date is not a claim",
        "that every branch or in-flight workflow adopted it immediately.",
        "",
        "[Build-step-only graph](openusd_build_timings_build_step_monthly.svg)",
        "",
        "## Filtering and interpretation",
        "",
        (
            "- Require completed job status and success conclusion, with positive"
            " elapsed time."
        ),
        "- If steps are present, require a successful Build USD step and no failed,",
        "  cancelled, or timed-out steps. A skipped cache-save step is permitted.",
        "- When steps are absent, retain successful job-level records; step-only stats",
        "  omit these jobs rather than treating missing steps as zero minutes.",
        "  Step-only and full-job statistics therefore use different sample sets.",
        (
            "- Deduplicate records with the same run ID, job name, start, and end"
            " timestamps."
        ),
        "  Real reruns with different execution times remain separate observations.",
        "- Do not trim fast or slow successful jobs just because of their duration.",
        (
            "  Successful short Windows jobs have successful build/test steps; cache"
            " effects"
        ),
        "  and changing workflow behavior may produce real timing variation.",
        "- Full-job elapsed time includes setup, build, test, and artifact handling,",
        "  but excludes queue time. It is not a compilation-only measurement.",
        "- Percentiles use inclusive linear interpolation at rank (n - 1) * p.",
        (
            "  Means and pooled percentiles weight individual executions, not months"
            " equally."
        ),
        "- Monthly bins use job start time in UTC. A low sample count makes quantiles",
        "  unstable; n=1 has identical p10, median, p90, and mean.",
        "",
        "| Filter outcome | Jobs |",
        "|---|---:|",
    ]
    lines += [f"| {reason} | {count} |" for reason, count in sorted(counts.items())]
    lines += [
        "",
        (
            "Included jobs lacking step detail:"
            f" {sum(not r['steps_available'] for r in included)}."
        ),
        "",
        "## Cost-model use",
        "",
        "The monthly Windows median changes sharply around June 2026, with both short",
        "and long successful executions afterward. A pooled historical average is not",
        (
            "necessarily representative of the current workflow. Linux durations also"
            " trend"
        ),
        "upward. Inspect monthly data before selecting a forward-looking duration.",
        "",
        "Wasm32 and Wasm64 must be budgeted as separate Linux-hosted build targets.",
        "Use the arithmetic mean for expected successful-build compute; p10/median/p90",
        "describe duration variability, not the low/mid/high trigger-growth scenarios.",
        "These hosted-runner measurements are not benchmarks of the proposed g6/Mac",
        "hardware. Failed attempts still consume compute even though this analysis",
        (
            "excludes them. Trigger forecasts include reruns, but successful-only"
            " durations"
        ),
        "do not by themselves estimate failure overhead or selective-rerun fan-out.",
        "Wasm configuration counts and activation date within Year 1 must be specified",
        "before replacing the existing native-only budget. Wasm artifact sizes also",
        "need separate measurement. The 16-configuration allocation is provisional.",
        "",
        "## Data and reproduction",
        "",
        "- [Monthly statistics CSV](openusd_build_timings_monthly.csv)",
        "- [Pooled statistics CSV](openusd_build_timings_summary.csv)",
        "- [Job-level timing and filter audit CSV](openusd_build_timings_jobs.csv)",
        "- Source: local Dolt workflow_jobs/workflow_runs, cached directly from the",
        (
            "  [GitHub Actions jobs"
            " API](https://docs.github.com/en/rest/actions/workflow-jobs)."
        ),
        "  GH Archive does not supply these timing records.",
        "",
        "```sh",
        "uv run python analyze_build_timings.py",
        "```",
        "",
    ]
    (output / "openusd_build_timings.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"filter_counts": counts, "overall": overall}, indent=2))


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--database-dir", type=Path, default=ROOT / ".cache/openusd_usage_dolt"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    try:
        analyze(args.database_dir, args.output_dir)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
