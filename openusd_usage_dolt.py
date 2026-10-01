#!/usr/bin/env python

"""Cache OpenUSD pull-request and GitHub Actions usage in Dolt and forecast demand.

The pull-request history comes from ClickHouse's public GitHub Archive mirror.
GitHub Actions workflows, runs, jobs, and steps come from the GitHub REST API
through the authenticated ``gh`` CLI.

Examples:
    ./run_openusd_usage.sh sync-all --start-date 2021-09-16
    ./run_openusd_usage.sh sync-releases
    ./run_openusd_usage.sh export-releases
    ./run_openusd_usage.sh report
    ./run_openusd_usage.sh sync-pr-events --start-date 2021-09-16
"""

import argparse
import collections
import csv
import datetime
import hashlib
import html
import json
import math
import re
import statistics
import subprocess
import sys
import traceback

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

###############################################################################
# Constants
###############################################################################


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE_DIR = THIS_DIR / ".cache" / "openusd_usage_dolt"
DEFAULT_REPORT = THIS_DIR / "openusd_usage_forecast.md"
DEFAULT_RELEASE_CSV = THIS_DIR / "openusd_triggering_events_by_release.csv"
DEFAULT_RELEASE_SVG = THIS_DIR / "openusd_triggering_events_by_release.svg"
DEFAULT_RELEASE_RATE_SVG = (
    THIS_DIR / "openusd_triggering_events_per_month_by_release.svg"
)
DEFAULT_COMPUTE_SVG = THIS_DIR / "openusd_runner_minutes_by_release.svg"
DEFAULT_COMPUTE_RATE_SVG = THIS_DIR / "openusd_runner_minutes_per_month_by_release.svg"
DEFAULT_FORECAST_CSV = THIS_DIR / "openusd_triggering_events_forecast.csv"
DEFAULT_EFFECTIVE_RELEASE_FORECAST_SVG = (
    THIS_DIR / "openusd_triggering_events_per_month_by_release_forecast.svg"
)
REPOSITORY = "PixarAnimationStudios/OpenUSD"
REPOSITORY_ID = 58168143
BUILD_WORKFLOW_NAME = "BuildUSD"
CLICKHOUSE_URL = "https://sql-clickhouse.clickhouse.com/"
CLICKHOUSE_USER = "demo"
FORECAST_MONTHS = 60
# The high scenario continues 100% of the observed per-release linear slope.
# The mid and low scenarios retain one-half and one-quarter of that slope to
# account for the small sample and the risk that the current release burst
# overstates durable growth.
FORECAST_SCENARIO_GROWTH = {
    "low": 0.25,
    "mid": 0.50,
    "high": 1.00,
}
TREND_COVERAGE_STATUSES = frozenset({"complete", "in_progress"})
# In the current release cycle, the first four weeks ran at about half the
# cycle-average rate, the middle period approached the average, and recent
# release-close weeks reached about 1.6 times the average. The three factors
# sum to 3.0, preserving the cycle average before the small trend adjustment;
# forecast generation normalizes each cycle to preserve it exactly.
RELEASE_PHASES = (
    ("early", 0.5),
    ("middle", 0.9),
    ("late", 1.6),
)
FORECAST_RELEASES = FORECAST_MONTHS // len(RELEASE_PHASES)
RELEASES_PER_YEAR = 4
FORECAST_YEARS = FORECAST_RELEASES // RELEASES_PER_YEAR
# Earlier cached months contain BuildUSD records, but complete all-workflow
# collection starts here. Treat earlier months as unavailable or partial.
COMPLETE_TRIGGER_COVERAGE_START = datetime.date(2025, 9, 1)
TRIGGER_CLUSTER_MINUTES = 5
SVG_CANVAS_WIDTH = 1600
SVG_CANVAS_HEIGHT = 820
STANDARD_MONTH_DAYS = 30.4375
LONG_RELEASE_CYCLE_FACTOR = 1.25
CADENCE_RELEASE_PATTERN = re.compile(r"^v\d{2}\.\d{2}$")
CORE_BUILD_JOB_FAMILIES = {
    "Linux": "linux",
    "macOS": "macos",
    "Windows": "windows",
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sync_metadata (
    meta_key VARCHAR(255) PRIMARY KEY,
    meta_value LONGTEXT NOT NULL,
    updated_at DATETIME(6) NOT NULL
);

CREATE TABLE IF NOT EXISTS releases (
    release_id BIGINT PRIMARY KEY,
    tag_name VARCHAR(64) NOT NULL,
    name VARCHAR(255) NULL,
    created_at DATETIME NULL,
    published_at DATETIME NOT NULL,
    html_url TEXT NULL,
    draft BOOLEAN NOT NULL,
    prerelease BOOLEAN NOT NULL,
    raw_json LONGTEXT NOT NULL,
    UNIQUE KEY idx_releases_tag_name (tag_name),
    KEY idx_releases_published_at (published_at)
);

CREATE TABLE IF NOT EXISTS pull_request_events (
    event_key CHAR(64) PRIMARY KEY,
    repo_id BIGINT NOT NULL,
    repo_name VARCHAR(255) NOT NULL,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NULL,
    action VARCHAR(64) NOT NULL,
    pr_number BIGINT NOT NULL,
    actor_login VARCHAR(255) NULL,
    creator_user_login VARCHAR(255) NULL,
    title TEXT NULL,
    state VARCHAR(32) NULL,
    closed_at DATETIME NULL,
    merged_at DATETIME NULL,
    head_ref TEXT NULL,
    head_sha VARCHAR(64) NULL,
    base_ref TEXT NULL,
    base_sha VARCHAR(64) NULL,
    merged BOOLEAN NOT NULL,
    commit_count BIGINT NOT NULL,
    additions BIGINT NOT NULL,
    deletions BIGINT NOT NULL,
    changed_files BIGINT NOT NULL,
    author_association VARCHAR(64) NULL,
    source VARCHAR(64) NOT NULL,
    raw_json LONGTEXT NOT NULL,
    KEY idx_pr_events_created_at (created_at),
    KEY idx_pr_events_action (action),
    KEY idx_pr_events_number (pr_number)
);

CREATE TABLE IF NOT EXISTS pull_request_timeline_events (
    event_key CHAR(64) PRIMARY KEY,
    pr_number BIGINT NOT NULL,
    event VARCHAR(64) NOT NULL,
    event_at DATETIME NULL,
    commit_sha VARCHAR(64) NULL,
    actor_login VARCHAR(255) NULL,
    source VARCHAR(64) NOT NULL,
    raw_json LONGTEXT NOT NULL,
    KEY idx_pr_timeline_event_at (event_at),
    KEY idx_pr_timeline_event (event),
    KEY idx_pr_timeline_number (pr_number)
);

CREATE TABLE IF NOT EXISTS workflows (
    workflow_id BIGINT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    path TEXT NOT NULL,
    state VARCHAR(64) NOT NULL,
    created_at DATETIME NULL,
    updated_at DATETIME NULL,
    html_url TEXT NULL,
    raw_json LONGTEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workflow_runs (
    run_id BIGINT PRIMARY KEY,
    workflow_id BIGINT NOT NULL,
    workflow_name VARCHAR(255) NOT NULL,
    run_number BIGINT NOT NULL,
    run_attempt BIGINT NOT NULL,
    event VARCHAR(64) NOT NULL,
    status VARCHAR(64) NULL,
    conclusion VARCHAR(64) NULL,
    created_at DATETIME NOT NULL,
    run_started_at DATETIME NULL,
    updated_at DATETIME NULL,
    head_branch TEXT NULL,
    head_sha VARCHAR(64) NULL,
    actor_login VARCHAR(255) NULL,
    html_url TEXT NULL,
    jobs_fetched_at DATETIME(6) NULL,
    raw_json LONGTEXT NOT NULL,
    KEY idx_runs_created_at (created_at),
    KEY idx_runs_workflow_event (workflow_id, event)
);

CREATE TABLE IF NOT EXISTS workflow_jobs (
    job_id BIGINT PRIMARY KEY,
    run_id BIGINT NOT NULL,
    workflow_name VARCHAR(255) NULL,
    name TEXT NOT NULL,
    status VARCHAR(64) NULL,
    conclusion VARCHAR(64) NULL,
    started_at DATETIME NULL,
    completed_at DATETIME NULL,
    runner_id BIGINT NULL,
    runner_name TEXT NULL,
    runner_group_id BIGINT NULL,
    runner_group_name TEXT NULL,
    labels_json LONGTEXT NOT NULL,
    raw_json LONGTEXT NOT NULL,
    KEY idx_jobs_run_id (run_id),
    KEY idx_jobs_started_at (started_at)
);

CREATE TABLE IF NOT EXISTS job_steps (
    job_id BIGINT NOT NULL,
    step_number BIGINT NOT NULL,
    name TEXT NOT NULL,
    status VARCHAR(64) NULL,
    conclusion VARCHAR(64) NULL,
    started_at DATETIME NULL,
    completed_at DATETIME NULL,
    raw_json LONGTEXT NOT NULL,
    PRIMARY KEY (job_id, step_number)
);
"""

###############################################################################
# Data structures
###############################################################################


###############################################################################
# Utility functions
###############################################################################


def run_command(
    args: Sequence[str],
    *,
    cwd: Path,
    input_text: str | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(args),
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=capture_output,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Command failed ({' '.join(args)}): {message}")
    return result


def parse_date(value: str) -> datetime.date:
    return datetime.date.fromisoformat(value)


def five_years_before(day: datetime.date) -> datetime.date:
    try:
        return day.replace(year=day.year - 5)
    except ValueError:
        return day.replace(year=day.year - 5, day=28)


def normalize_datetime(value: str | None) -> str:
    if not value or value.startswith("1970-01-01"):
        return ""
    return value.replace("T", " ").replace("Z", "").split(".", 1)[0]


def first_of_month(day: datetime.date) -> datetime.date:
    return day.replace(day=1)


def next_month(day: datetime.date) -> datetime.date:
    if day.month == 12:
        return datetime.date(day.year + 1, 1, 1)
    return datetime.date(day.year, day.month + 1, 1)


def iter_months(start: datetime.date, end: datetime.date) -> Iterable[datetime.date]:
    month = first_of_month(start)
    while month < end:
        yield month
        month = next_month(month)


def json_compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stable_event_key(event: dict) -> str:
    fields = (
        event.get("repo_id"),
        event.get("event_type"),
        event.get("created_at"),
        event.get("action"),
        event.get("number"),
        event.get("head_sha"),
        event.get("actor_login"),
    )
    return hashlib.sha256(json_compact(fields).encode("utf-8")).hexdigest()


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


###############################################################################
# Dolt functions
###############################################################################


def initialize_database(database_dir: Path):
    database_dir.mkdir(parents=True, exist_ok=True)
    if not (database_dir / ".dolt").exists():
        run_command(
            [
                "dolt",
                "init",
                "--name",
                "OpenUSD Usage Import",
                "--email",
                "openusd-usage@localhost",
            ],
            cwd=database_dir,
        )
        run_command(
            [
                "dolt",
                "config",
                "--local",
                "--add",
                "user.name",
                "OpenUSD Usage Import",
            ],
            cwd=database_dir,
        )
        run_command(
            [
                "dolt",
                "config",
                "--local",
                "--add",
                "user.email",
                "openusd-usage@localhost",
            ],
            cwd=database_dir,
        )
    run_command(["dolt", "sql"], cwd=database_dir, input_text=SCHEMA_SQL)


def import_csv(database_dir: Path, table: str, path: Path):
    run_command(
        ["dolt", "table", "import", "-u", "--quiet", table, str(path)],
        cwd=database_dir,
    )


def commit_database(database_dir: Path, message: str):
    run_command(["dolt", "add", "-A"], cwd=database_dir)
    run_command(["dolt", "commit", "--skip-empty", "-m", message], cwd=database_dir)


def query_csv(database_dir: Path, query: str) -> list[dict[str, str]]:
    result = run_command(["dolt", "sql", "-r", "csv", "-q", query], cwd=database_dir)
    return list(csv.DictReader(result.stdout.splitlines()))


def import_metadata(database_dir: Path, values: dict[str, str], stage_dir: Path):
    now = (
        datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(" ")
    )
    rows = [
        {"meta_key": key, "meta_value": value, "updated_at": now}
        for key, value in sorted(values.items())
    ]
    path = stage_dir / "sync_metadata.csv"
    write_csv(path, ("meta_key", "meta_value", "updated_at"), rows)
    import_csv(database_dir, "sync_metadata", path)


###############################################################################
# Pull-request event collection
###############################################################################


def fetch_pull_request_events(
    start_date: datetime.date, end_date: datetime.date
) -> tuple[list[dict], str]:
    columns = (
        "file_time,event_type,actor_login,repo_name,repo_id,created_at,updated_at,"
        "action,creator_user_login,number,title,state,author_association,closed_at,"
        "merged_at,head_ref,head_sha,base_ref,base_sha,merged,commits,additions,"
        "deletions,changed_files"
    )
    query = f"""
SELECT {columns}
FROM git.github_events
WHERE repo_id = '{REPOSITORY_ID}'
  AND event_type = 'PullRequestEvent'
  AND created_at >= '{start_date.isoformat()}'
  AND created_at < '{end_date.isoformat()}'
ORDER BY created_at, number, action
FORMAT JSONEachRow
""".strip()
    response = requests.post(
        CLICKHOUSE_URL,
        auth=(CLICKHOUSE_USER, ""),
        data=query.encode("utf-8"),
        timeout=90,
    )
    response.raise_for_status()
    events = [json.loads(line) for line in response.text.splitlines() if line]
    return events, query


def pull_request_event_rows(events: Iterable[dict]) -> Iterable[dict]:
    for event in events:
        yield {
            "event_key": stable_event_key(event),
            "repo_id": event["repo_id"],
            "repo_name": event["repo_name"],
            "created_at": normalize_datetime(event["created_at"]),
            "updated_at": normalize_datetime(event.get("updated_at")),
            "action": event["action"],
            "pr_number": event["number"],
            "actor_login": event.get("actor_login", ""),
            "creator_user_login": event.get("creator_user_login", ""),
            "title": event.get("title", ""),
            "state": event.get("state", ""),
            "closed_at": normalize_datetime(event.get("closed_at")),
            "merged_at": normalize_datetime(event.get("merged_at")),
            "head_ref": event.get("head_ref", ""),
            "head_sha": event.get("head_sha", ""),
            "base_ref": event.get("base_ref", ""),
            "base_sha": event.get("base_sha", ""),
            "merged": int(bool(event.get("merged"))),
            "commit_count": event.get("commits", 0),
            "additions": event.get("additions", 0),
            "deletions": event.get("deletions", 0),
            "changed_files": event.get("changed_files", 0),
            "author_association": event.get("author_association", ""),
            "source": "clickhouse-github-archive",
            "raw_json": json_compact(event),
        }


def sync_pull_request_events(
    database_dir: Path,
    stage_dir: Path,
    start_date: datetime.date,
    end_date: datetime.date,
):
    print(f"Fetching pull-request events from {start_date} through {end_date}...")
    events, query = fetch_pull_request_events(start_date, end_date)
    fieldnames = (
        "event_key",
        "repo_id",
        "repo_name",
        "created_at",
        "updated_at",
        "action",
        "pr_number",
        "actor_login",
        "creator_user_login",
        "title",
        "state",
        "closed_at",
        "merged_at",
        "head_ref",
        "head_sha",
        "base_ref",
        "base_sha",
        "merged",
        "commit_count",
        "additions",
        "deletions",
        "changed_files",
        "author_association",
        "source",
        "raw_json",
    )
    path = stage_dir / "pull_request_events.csv"
    write_csv(path, fieldnames, pull_request_event_rows(events))
    import_csv(database_dir, "pull_request_events", path)
    import_metadata(
        database_dir,
        {
            "pr_events_end_date_exclusive": end_date.isoformat(),
            "pr_events_query_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "pr_events_row_count": str(len(events)),
            "pr_events_source": CLICKHOUSE_URL,
            "pr_events_start_date": start_date.isoformat(),
        },
        stage_dir,
    )
    commit_database(
        database_dir,
        f"Import {len(events)} OpenUSD pull-request events through {end_date}",
    )
    print(f"Imported {len(events)} pull-request events.")


###############################################################################
# GitHub Actions collection
###############################################################################


def github_api(endpoint: str, params: dict[str, object] | None = None) -> dict | list:
    args = ["gh", "api", endpoint, "--method", "GET"]
    for key, value in sorted((params or {}).items()):
        args.extend(("-f", f"{key}={value}"))
    result = run_command(args, cwd=THIS_DIR)
    return json.loads(result.stdout)


def github_paginated(
    endpoint: str, key: str, params: dict[str, object] | None = None
) -> list[dict]:
    rows = []
    page = 1
    while True:
        page_params = dict(params or {})
        page_params.update({"page": page, "per_page": 100})
        payload = github_api(endpoint, page_params)
        page_rows = payload.get(key, [])
        rows.extend(page_rows)
        if len(page_rows) < 100:
            break
        if page >= 10 and payload.get("total_count", 0) > 1000:
            raise RuntimeError(
                f"GitHub capped {endpoint} at 1,000 results; use smaller date windows"
            )
        page += 1
    return rows


def github_list_paginated(endpoint: str) -> list[dict]:
    rows = []
    page = 1
    while True:
        page_rows = github_api(endpoint, {"page": page, "per_page": 100})
        if not isinstance(page_rows, list):
            raise TypeError(f"Expected a list response from {endpoint}")
        rows.extend(page_rows)
        if len(page_rows) < 100:
            return rows
        page += 1


def release_rows(releases: Iterable[dict]) -> Iterable[dict]:
    for release in releases:
        yield {
            "release_id": release["id"],
            "tag_name": release["tag_name"],
            "name": release.get("name", ""),
            "created_at": normalize_datetime(release.get("created_at")),
            "published_at": normalize_datetime(release.get("published_at")),
            "html_url": release.get("html_url", ""),
            "draft": release.get("draft", False),
            "prerelease": release.get("prerelease", False),
            "raw_json": json_compact(release),
        }


def sync_releases(database_dir: Path, stage_dir: Path):
    releases = github_list_paginated(f"repos/{REPOSITORY}/releases")
    release_path = stage_dir / "releases.csv"
    release_fields = (
        "release_id",
        "tag_name",
        "name",
        "created_at",
        "published_at",
        "html_url",
        "draft",
        "prerelease",
        "raw_json",
    )
    write_csv(release_path, release_fields, release_rows(releases))
    import_csv(database_dir, "releases", release_path)
    import_metadata(
        database_dir,
        {
            "release_count": str(len(releases)),
            "release_source": "GitHub REST API via gh",
        },
        stage_dir,
    )
    commit_database(database_dir, f"Import {len(releases)} OpenUSD releases")
    print(f"Imported {len(releases)} releases.")


def timeline_event_at(event: dict) -> str:
    for key in ("created_at", "submitted_at"):
        if event.get(key):
            return normalize_datetime(event[key])
    for key in ("committer", "author"):
        if (event.get(key) or {}).get("date"):
            return normalize_datetime(event[key]["date"])
    return ""


def timeline_event_rows(
    events_by_pr: Iterable[tuple[int, list[dict]]],
) -> Iterable[dict]:
    for pr_number, events in events_by_pr:
        for index, event in enumerate(events):
            identity = (
                pr_number,
                event.get("event"),
                event.get("id"),
                event.get("node_id"),
                event.get("sha"),
                index,
            )
            yield {
                "event_key": (
                    hashlib.sha256(json_compact(identity).encode("utf-8")).hexdigest()
                ),
                "pr_number": pr_number,
                "event": event.get("event", "unknown"),
                "event_at": timeline_event_at(event),
                "commit_sha": event.get("sha") or event.get("commit_id", ""),
                "actor_login": (event.get("actor") or {}).get("login", ""),
                "source": "github-issue-timeline-api",
                "raw_json": json_compact(event),
            }


def sync_pull_request_timelines(database_dir: Path, stage_dir: Path):
    pr_rows = query_csv(
        database_dir,
        "SELECT DISTINCT pr_number FROM pull_request_events ORDER BY pr_number;",
    )
    pr_numbers = [int(row["pr_number"]) for row in pr_rows]

    def fetch_timeline(pr_number: int) -> tuple[int, list[dict]]:
        endpoint = f"repos/{REPOSITORY}/issues/{pr_number}/timeline"
        return pr_number, github_list_paginated(endpoint)

    events_by_pr = []
    print(f"Fetching timelines for {len(pr_numbers)} pull requests...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_timeline, number) for number in pr_numbers]
        for index, future in enumerate(as_completed(futures), 1):
            events_by_pr.append(future.result())
            if index % 50 == 0 or index == len(pr_numbers):
                print(f"  fetched timelines for {index}/{len(pr_numbers)} PRs")

    fields = (
        "event_key",
        "pr_number",
        "event",
        "event_at",
        "commit_sha",
        "actor_login",
        "source",
        "raw_json",
    )
    rows = list(timeline_event_rows(events_by_pr))
    path = stage_dir / "pull_request_timeline_events.csv"
    write_csv(path, fields, rows)
    import_csv(database_dir, "pull_request_timeline_events", path)
    import_metadata(
        database_dir,
        {
            "pr_timeline_event_count": str(len(rows)),
            "pr_timeline_pr_count": str(len(pr_numbers)),
            "pr_timeline_source": "GitHub issue timeline REST API via gh",
        },
        stage_dir,
    )
    commit_database(
        database_dir,
        f"Import timelines for {len(pr_numbers)} OpenUSD pull requests",
    )
    print(f"Imported {len(rows)} timeline events for {len(pr_numbers)} PRs.")


def workflow_rows(workflows: Iterable[dict]) -> Iterable[dict]:
    for workflow in workflows:
        yield {
            "workflow_id": workflow["id"],
            "name": workflow["name"],
            "path": workflow["path"],
            "state": workflow["state"],
            "created_at": normalize_datetime(workflow.get("created_at")),
            "updated_at": normalize_datetime(workflow.get("updated_at")),
            "html_url": workflow.get("html_url", ""),
            "raw_json": json_compact(workflow),
        }


def workflow_run_rows(runs: Iterable[dict], fetched_at: str) -> Iterable[dict]:
    for run in runs:
        yield {
            "run_id": run["id"],
            "workflow_id": run["workflow_id"],
            "workflow_name": run.get("name", ""),
            "run_number": run["run_number"],
            "run_attempt": run.get("run_attempt", 1),
            "event": run["event"],
            "status": run.get("status", ""),
            "conclusion": run.get("conclusion", ""),
            "created_at": normalize_datetime(run["created_at"]),
            "run_started_at": normalize_datetime(run.get("run_started_at")),
            "updated_at": normalize_datetime(run.get("updated_at")),
            "head_branch": run.get("head_branch", ""),
            "head_sha": run.get("head_sha", ""),
            "actor_login": (run.get("actor") or {}).get("login", ""),
            "html_url": run.get("html_url", ""),
            "jobs_fetched_at": fetched_at,
            "raw_json": json_compact(run),
        }


def job_rows(jobs: Iterable[dict]) -> Iterable[dict]:
    for job in jobs:
        yield {
            "job_id": job["id"],
            "run_id": job["run_id"],
            "workflow_name": job.get("workflow_name", ""),
            "name": job["name"],
            "status": job.get("status", ""),
            "conclusion": job.get("conclusion", ""),
            "started_at": normalize_datetime(job.get("started_at")),
            "completed_at": normalize_datetime(job.get("completed_at")),
            "runner_id": job.get("runner_id", ""),
            "runner_name": job.get("runner_name", ""),
            "runner_group_id": job.get("runner_group_id", ""),
            "runner_group_name": job.get("runner_group_name", ""),
            "labels_json": json_compact(job.get("labels", [])),
            "raw_json": json_compact(job),
        }


def step_rows(jobs: Iterable[dict]) -> Iterable[dict]:
    for job in jobs:
        for step in job.get("steps", []):
            yield {
                "job_id": job["id"],
                "step_number": step["number"],
                "name": step["name"],
                "status": step.get("status", ""),
                "conclusion": step.get("conclusion", ""),
                "started_at": normalize_datetime(step.get("started_at")),
                "completed_at": normalize_datetime(step.get("completed_at")),
                "raw_json": json_compact(step),
            }


def fetch_action_runs(start_date: datetime.date, end_date: datetime.date) -> list[dict]:
    runs = []
    for month in iter_months(start_date, end_date):
        window_start = max(start_date, month)
        window_end = min(end_date, next_month(month))
        last_day = window_end - datetime.timedelta(days=1)
        created = f"{window_start.isoformat()}..{last_day.isoformat()}"
        window_runs = github_paginated(
            f"repos/{REPOSITORY}/actions/runs",
            "workflow_runs",
            {"created": created},
        )
        print(f"  {created}: {len(window_runs)} workflow runs")
        runs.extend(window_runs)
    return runs


def sync_actions(
    database_dir: Path,
    stage_dir: Path,
    start_date: datetime.date,
    end_date: datetime.date,
):
    workflow_payload = github_api(f"repos/{REPOSITORY}/actions/workflows")
    workflows = workflow_payload["workflows"]
    workflow_path = stage_dir / "workflows.csv"
    workflow_fields = (
        "workflow_id",
        "name",
        "path",
        "state",
        "created_at",
        "updated_at",
        "html_url",
        "raw_json",
    )
    write_csv(workflow_path, workflow_fields, workflow_rows(workflows))
    import_csv(database_dir, "workflows", workflow_path)

    print(f"Fetching all Actions runs from {start_date} through {end_date}...")
    runs = fetch_action_runs(start_date, end_date)
    fetched_at = (
        datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(" ")
    )
    run_fields = (
        "run_id",
        "workflow_id",
        "workflow_name",
        "run_number",
        "run_attempt",
        "event",
        "status",
        "conclusion",
        "created_at",
        "run_started_at",
        "updated_at",
        "head_branch",
        "head_sha",
        "actor_login",
        "html_url",
        "jobs_fetched_at",
        "raw_json",
    )
    run_path = stage_dir / "workflow_runs.csv"
    write_csv(run_path, run_fields, workflow_run_rows(runs, ""))
    import_csv(database_dir, "workflow_runs", run_path)
    commit_database(database_dir, f"Import {len(runs)} OpenUSD Actions runs")

    def fetch_jobs(run: dict) -> list[dict]:
        return github_paginated(
            f"repos/{REPOSITORY}/actions/runs/{run['id']}/jobs",
            "jobs",
            {"filter": "all"},
        )

    jobs = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_jobs, run): run["id"] for run in runs}
        for index, future in enumerate(as_completed(futures), 1):
            jobs.extend(future.result())
            if index % 50 == 0 or index == len(runs):
                print(f"  fetched jobs for {index}/{len(runs)} runs")

    job_fields = (
        "job_id",
        "run_id",
        "workflow_name",
        "name",
        "status",
        "conclusion",
        "started_at",
        "completed_at",
        "runner_id",
        "runner_name",
        "runner_group_id",
        "runner_group_name",
        "labels_json",
        "raw_json",
    )
    step_fields = (
        "job_id",
        "step_number",
        "name",
        "status",
        "conclusion",
        "started_at",
        "completed_at",
        "raw_json",
    )
    job_path = stage_dir / "workflow_jobs.csv"
    step_path = stage_dir / "job_steps.csv"
    write_csv(job_path, job_fields, job_rows(jobs))
    write_csv(step_path, step_fields, step_rows(jobs))
    write_csv(run_path, run_fields, workflow_run_rows(runs, fetched_at))
    import_csv(database_dir, "workflow_runs", run_path)
    import_csv(database_dir, "workflow_jobs", job_path)
    import_csv(database_dir, "job_steps", step_path)
    import_metadata(
        database_dir,
        {
            "actions_end_date_exclusive": end_date.isoformat(),
            "actions_job_count": str(len(jobs)),
            "actions_run_count": str(len(runs)),
            "actions_source": "GitHub REST API via gh",
            "actions_start_date": start_date.isoformat(),
            "actions_workflow_count": str(len(workflows)),
        },
        stage_dir,
    )
    commit_database(
        database_dir,
        f"Import {len(runs)} OpenUSD Actions runs and {len(jobs)} jobs",
    )
    print(f"Imported {len(runs)} runs and {len(jobs)} jobs.")


###############################################################################
# Trigger-event and compute exports
###############################################################################


def workflow_run_export_rows(
    database_dir: Path, start: datetime.date, end: datetime.date
) -> list[dict[str, str]]:
    return query_csv(
        database_dir,
        "SELECT run_id, workflow_id, workflow_name, event, created_at, "
        "head_branch, head_sha, actor_login, raw_json FROM workflow_runs "
        f"WHERE created_at >= '{start}' AND created_at < '{end}' "
        "ORDER BY created_at, run_id;",
    )


def workflow_job_export_rows(
    database_dir: Path, start: datetime.date, end: datetime.date
) -> list[dict[str, str]]:
    return query_csv(
        database_dir,
        "SELECT j.job_id, j.run_id, r.workflow_name, j.name, j.started_at, "
        "j.completed_at, j.labels_json, j.conclusion, j.raw_json "
        "FROM workflow_jobs j "
        "JOIN workflow_runs r ON r.run_id = j.run_id "
        f"WHERE r.created_at >= '{start}' AND r.created_at < '{end}' "
        "ORDER BY j.started_at, j.job_id;",
    )


def parse_database_datetime(value: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(value)


def source_signature(run: dict[str, str]) -> tuple[str, str]:
    event = run["event"]
    head_sha = run["head_sha"]
    if event in {
        "merge_group",
        "pull_request",
        "pull_request_target",
        "push",
        "workflow_run",
    }:
        identity = head_sha or f"{run['head_branch']}:{run['actor_login']}"
        return "source", identity
    if event == "schedule":
        created_at = parse_database_datetime(run["created_at"])
        return "schedule", created_at.strftime("%Y-%m-%dT%H:%M")
    return "run", run["run_id"]


def clustered_events(
    occurrences: Iterable[dict], signature_key: str, time_key: str
) -> list[dict]:
    clusters = []
    latest_by_signature: dict[object, dict] = {}
    maximum_gap = datetime.timedelta(minutes=TRIGGER_CLUSTER_MINUTES)
    for occurrence in sorted(occurrences, key=lambda item: item[time_key]):
        signature = occurrence[signature_key]
        previous = latest_by_signature.get(signature)
        if previous and occurrence[time_key] - previous["latest_at"] <= maximum_gap:
            previous["latest_at"] = occurrence[time_key]
            previous["occurrences"].append(occurrence)
            continue
        cluster = {
            "started_at": occurrence[time_key],
            "latest_at": occurrence[time_key],
            "occurrences": [occurrence],
        }
        clusters.append(cluster)
        latest_by_signature[signature] = cluster
    return clusters


def infer_source_events(run_rows: Iterable[dict[str, str]]) -> list[dict]:
    occurrences = []
    for run in run_rows:
        occurrences.append(
            {
                "created_at": parse_database_datetime(run["created_at"]),
                "event": run["event"],
                "run_id": int(run["run_id"]),
                "signature": source_signature(run),
                "workflow_id": int(run["workflow_id"]),
                "workflow_name": run["workflow_name"],
            }
        )
    return clustered_events(occurrences, "signature", "created_at")


def job_attempt(job: dict[str, str]) -> int:
    return int(json.loads(job["raw_json"]).get("run_attempt", 1))


def infer_rerun_events(
    run_rows: Iterable[dict[str, str]], job_rows: Iterable[dict[str, str]]
) -> tuple[list[dict], list[dict]]:
    runs_by_id = {int(run["run_id"]): run for run in run_rows}
    attempts: dict[tuple[int, int], dict] = {}
    for job in job_rows:
        attempt = job_attempt(job)
        if attempt <= 1 or not job["started_at"]:
            continue
        run_id = int(job["run_id"])
        started_at = parse_database_datetime(job["started_at"])
        key = (run_id, attempt)
        current = attempts.get(key)
        if current is None or started_at < current["started_at"]:
            run = runs_by_id[run_id]
            attempts[key] = {
                "attempt": attempt,
                "run_id": run_id,
                "signature": (
                    run["head_sha"] or run["head_branch"],
                    attempt,
                ),
                "started_at": started_at,
                "workflow_id": int(run["workflow_id"]),
                "workflow_name": run["workflow_name"],
            }
    attempt_rows = list(attempts.values())
    return (
        clustered_events(attempt_rows, "signature", "started_at"),
        attempt_rows,
    )


def core_build_family(job: dict[str, str]) -> str | None:
    if job["workflow_name"] != BUILD_WORKFLOW_NAME:
        return None
    return CORE_BUILD_JOB_FAMILIES.get(job["name"])


def job_minutes(job: dict[str, str]) -> float:
    if not job["started_at"] or not job["completed_at"]:
        return 0.0
    started_at = parse_database_datetime(job["started_at"])
    completed_at = parse_database_datetime(job["completed_at"])
    return max(0.0, (completed_at - started_at).total_seconds() / 60)


def cached_cadence_releases(database_dir: Path) -> list[dict[str, str]]:
    rows = query_csv(
        database_dir,
        "SELECT tag_name, name, published_at, html_url FROM releases "
        "WHERE draft = FALSE AND prerelease = FALSE ORDER BY published_at;",
    )
    releases = [
        row for row in rows if CADENCE_RELEASE_PATTERN.fullmatch(row["tag_name"])
    ]
    if len(releases) < 2:
        raise RuntimeError("At least two official cadence releases must be cached")
    return releases


def release_cycles(
    releases: Sequence[dict[str, str]],
    first_observed: datetime.datetime,
    data_end: datetime.datetime,
) -> list[dict]:
    release_times = [
        parse_database_datetime(release["published_at"]) for release in releases
    ]
    durations = [
        (end - start).total_seconds() / 86400.0
        for start, end in zip(release_times, release_times[1:])
    ]
    long_cycle_days = statistics.median(durations) * LONG_RELEASE_CYCLE_FACTOR
    complete_start = datetime.datetime.combine(
        COMPLETE_TRIGGER_COVERAGE_START, datetime.time.min
    )
    cycles = []
    for previous, target, start, end, duration_days in zip(
        releases, releases[1:], release_times, release_times[1:], durations
    ):
        if end <= first_observed or start >= data_end:
            continue
        cycles.append(
            {
                "release_cycle": target["tag_name"],
                "previous_release": previous["tag_name"],
                "release_url": target["html_url"],
                "cycle_start": start,
                "cycle_end": min(end, data_end),
                "cycle_days": min(
                    duration_days, (data_end - start).total_seconds() / 86400.0
                ),
                "coverage_status": "complete" if start >= complete_start else "partial",
                "significantly_long_cycle": duration_days > long_cycle_days,
            }
        )
    latest_release = releases[-1]
    latest_time = release_times[-1]
    if latest_time < data_end:
        cycles.append(
            {
                "release_cycle": f"post-{latest_release['tag_name']}",
                "previous_release": latest_release["tag_name"],
                "release_url": "",
                "cycle_start": latest_time,
                "cycle_end": data_end,
                "cycle_days": (data_end - latest_time).total_seconds() / 86400.0,
                "coverage_status": (
                    "in_progress" if latest_time >= complete_start else "partial"
                ),
                "significantly_long_cycle": False,
            }
        )
    return cycles


def cycle_for_time(cycles: Sequence[dict], value: datetime.datetime) -> dict | None:
    for cycle in cycles:
        if cycle["cycle_start"] <= value < cycle["cycle_end"]:
            return cycle
    return None


def cached_action_end(database_dir: Path) -> datetime.datetime:
    rows = query_csv(
        database_dir,
        "SELECT meta_value FROM sync_metadata "
        "WHERE meta_key = 'actions_end_date_exclusive';",
    )
    if not rows:
        raise RuntimeError("Actions sync metadata is missing")
    return datetime.datetime.fromisoformat(rows[0]["meta_value"])


def release_triggering_event_rows(database_dir: Path) -> list[dict]:
    releases = cached_cadence_releases(database_dir)
    data_end = cached_action_end(database_dir)
    first_rows = query_csv(
        database_dir,
        "SELECT MIN(created_at) AS first_observed FROM workflow_runs;",
    )
    if not first_rows or not first_rows[0]["first_observed"]:
        raise RuntimeError("No workflow runs are cached")
    first_observed = parse_database_datetime(first_rows[0]["first_observed"])
    cycles = release_cycles(releases, first_observed, data_end)
    start = min(cycle["cycle_start"] for cycle in cycles).date()
    end = data_end.date()
    run_rows = workflow_run_export_rows(database_dir, start, end)
    job_rows = workflow_job_export_rows(database_dir, start, end)
    source_events = infer_source_events(run_rows)
    rerun_events, rerun_attempts = infer_rerun_events(run_rows, job_rows)

    for cycle in cycles:
        cycle.update(
            {
                "source_count": 0,
                "rerun_count": 0,
                "rerun_attempt_count": 0,
                "run_count": 0,
                "minutes": {"linux": 0.0, "macos": 0.0, "windows": 0.0},
            }
        )
    for event in source_events:
        cycle = cycle_for_time(cycles, event["started_at"])
        if cycle:
            cycle["source_count"] += 1
    for event in rerun_events:
        cycle = cycle_for_time(cycles, event["started_at"])
        if cycle:
            cycle["rerun_count"] += 1
    for attempt in rerun_attempts:
        cycle = cycle_for_time(cycles, attempt["started_at"])
        if cycle:
            cycle["rerun_attempt_count"] += 1
    for run in run_rows:
        cycle = cycle_for_time(cycles, parse_database_datetime(run["created_at"]))
        if cycle:
            cycle["run_count"] += 1
    for job in job_rows:
        family = core_build_family(job)
        if not family or not job["started_at"]:
            continue
        cycle = cycle_for_time(cycles, parse_database_datetime(job["started_at"]))
        if cycle:
            cycle["minutes"][family] += job_minutes(job)

    rows = []
    for cycle in cycles:
        source_count = cycle["source_count"]
        effective_count = source_count + cycle["rerun_count"]
        standard_months = cycle["cycle_days"] / STANDARD_MONTH_DAYS
        minutes = cycle["minutes"]
        rows.append(
            {
                "release_cycle": cycle["release_cycle"],
                "previous_release": cycle["previous_release"],
                "release_url": cycle["release_url"],
                "cycle_start_utc": cycle["cycle_start"].isoformat(" "),
                "cycle_end_utc": cycle["cycle_end"].isoformat(" "),
                "coverage_status": cycle["coverage_status"],
                "significantly_long_cycle": cycle["significantly_long_cycle"],
                "cycle_days": f"{cycle['cycle_days']:.2f}",
                "cycle_months_30_4375_days": f"{standard_months:.3f}",
                "source_triggering_events": source_count,
                "rerun_triggering_events": cycle["rerun_count"],
                "triggering_events_including_reruns": effective_count,
                "workflow_runs": cycle["run_count"],
                "workflow_rerun_attempts": cycle["rerun_attempt_count"],
                "source_events_per_30_4375_days": (
                    f"{source_count / standard_months:.1f}"
                ),
                "events_including_reruns_per_30_4375_days": (
                    f"{effective_count / standard_months:.1f}"
                ),
                "linux_x64_runner_minutes": f"{minutes['linux']:.1f}",
                "macos_runner_minutes": f"{minutes['macos']:.1f}",
                "windows_runner_minutes": f"{minutes['windows']:.1f}",
                "linux_x64_minutes_per_30_4375_days": (
                    f"{minutes['linux'] / standard_months:.1f}"
                ),
                "macos_minutes_per_30_4375_days": (
                    f"{minutes['macos'] / standard_months:.1f}"
                ),
                "windows_minutes_per_30_4375_days": (
                    f"{minutes['windows'] / standard_months:.1f}"
                ),
                "linux_x64_minutes_per_source_event": (
                    f"{minutes['linux'] / source_count:.1f}" if source_count else ""
                ),
                "macos_minutes_per_source_event": (
                    f"{minutes['macos'] / source_count:.1f}" if source_count else ""
                ),
                "windows_minutes_per_source_event": (
                    f"{minutes['windows'] / source_count:.1f}" if source_count else ""
                ),
            }
        )
    return rows


def svg_text(
    x: float,
    y: float,
    value: str,
    *,
    css_class: str = "",
    anchor: str = "start",
    transform: str = "",
) -> str:
    attributes = [f'x="{x:.1f}"', f'y="{y:.1f}"', f'text-anchor="{anchor}"']
    if css_class:
        attributes.append(f'class="{css_class}"')
    if transform:
        attributes.append(f'transform="{transform}"')
    return f"<text {' '.join(attributes)}>{html.escape(value)}</text>"


def nice_axis_max(maximum: float) -> tuple[float, float]:
    if maximum <= 50:
        step = 10.0
    elif maximum <= 100:
        step = 20.0
    elif maximum <= 500:
        step = 100.0
    elif maximum <= 2000:
        step = 500.0
    elif maximum <= 10000:
        step = 2000.0
    else:
        step = 10000.0
    return math.ceil(maximum / step) * step, step


def svg_document(title: str, subtitle: str, body: Iterable[str]) -> str:
    return "\n".join(
        [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_CANVAS_WIDTH}" '
                f'height="{SVG_CANVAS_HEIGHT}" '
                f'viewBox="0 0 {SVG_CANVAS_WIDTH} {SVG_CANVAS_HEIGHT}" role="img">'
            ),
            f"<title>{html.escape(title)}</title>",
            f"<desc>{html.escape(subtitle)}</desc>",
            "<style>",
            "text { font-family: Arial, sans-serif; fill: #24292f; }",
            ".title { font-size: 28px; font-weight: 700; }",
            ".subtitle { font-size: 16px; fill: #57606a; }",
            ".axis { font-size: 13px; fill: #57606a; }",
            ".value { font-size: 11px; font-weight: 700; }",
            ".note { font-size: 14px; fill: #57606a; }",
            ".grid { stroke: #d0d7de; stroke-width: 1; }",
            ".axis-line { stroke: #57606a; stroke-width: 1.5; }",
            "</style>",
            (
                f'<rect width="{SVG_CANVAS_WIDTH}" height="{SVG_CANVAS_HEIGHT}" '
                'fill="#ffffff" />'
            ),
            *body,
            "</svg>",
            "",
        ]
    )


def render_coverage_background(
    rows: Sequence[dict],
    plot_left: float,
    plot_top: float,
    plot_width: float,
    plot_height: float,
) -> list[str]:
    cell_width = plot_width / len(rows)
    output = []
    for index, row in enumerate(rows):
        if row["coverage_status"] == "complete":
            continue
        fill = "#ddf4ff" if row["coverage_status"] == "in_progress" else "#fff8c5"
        output.append(
            f'<rect x="{plot_left + index * cell_width:.1f}" y="{plot_top:.1f}" '
            f'width="{cell_width + 0.2:.1f}" height="{plot_height:.1f}" '
            f'fill="{fill}" />'
        )
    return output


def render_release_axes(
    rows: Sequence[dict],
    axis_max: float,
    axis_step: float,
    plot_left: float,
    plot_top: float,
    plot_width: float,
    plot_height: float,
) -> list[str]:
    output = []
    tick = 0.0
    while tick <= axis_max + 0.001:
        y = plot_top + plot_height - tick / axis_max * plot_height
        output.append(
            f'<line x1="{plot_left:.1f}" y1="{y:.1f}" '
            f'x2="{plot_left + plot_width:.1f}" y2="{y:.1f}" class="grid" />'
        )
        output.append(
            svg_text(
                plot_left - 12, y + 4, f"{tick:,.0f}", css_class="axis", anchor="end"
            )
        )
        tick += axis_step
    output.append(
        f'<line x1="{plot_left:.1f}" y1="{plot_top + plot_height:.1f}" '
        f'x2="{plot_left + plot_width:.1f}" y2="{plot_top + plot_height:.1f}" '
        'class="axis-line" />'
    )
    cell_width = plot_width / len(rows)
    for index, row in enumerate(rows):
        x = plot_left + (index + 0.5) * cell_width
        output.append(
            svg_text(
                x,
                plot_top + plot_height + 24,
                row["release_cycle"],
                css_class="axis",
                anchor="middle",
            )
        )
        output.append(
            svg_text(
                x,
                plot_top + plot_height + 43,
                f"{float(row['cycle_months_30_4375_days']):.2f} months",
                css_class="axis",
                anchor="middle",
            )
        )
    return output


def render_trigger_count_svg(
    rows: Sequence[dict], path: Path, *, per_standard_month: bool
):
    source_key = (
        "source_events_per_30_4375_days"
        if per_standard_month
        else "source_triggering_events"
    )
    effective_key = (
        "events_including_reruns_per_30_4375_days"
        if per_standard_month
        else "triggering_events_including_reruns"
    )
    values = [float(row[effective_key]) for row in rows if row[effective_key] != ""]
    axis_max, axis_step = nice_axis_max(max(values, default=1))
    plot_left, plot_top = 90.0, 145.0
    plot_width, plot_height = 1460.0, 480.0
    cell_width = plot_width / len(rows)
    body = [
        svg_text(
            55,
            52,
            (
                "Monthly Trigger Rate by Release (30.4375 Days)"
                if per_standard_month
                else "OpenUSD Triggering Events by Release"
            ),
            css_class="title",
        ),
        svg_text(
            55,
            82,
            (
                "Release-cycle totals normalized by exact elapsed days."
                if per_standard_month
                else "Cycle totals; the second bar adds inferred rerun events."
            ),
            css_class="subtitle",
        ),
        *render_coverage_background(rows, plot_left, plot_top, plot_width, plot_height),
        *render_release_axes(
            rows, axis_max, axis_step, plot_left, plot_top, plot_width, plot_height
        ),
    ]
    for index, row in enumerate(rows):
        if row[source_key] == "":
            continue
        source = float(row[source_key])
        effective = float(row[effective_key])
        group_left = plot_left + index * cell_width + cell_width * 0.12
        bar_width = cell_width * 0.34
        for offset, value, fill in (
            (0.0, source, "#0969da"),
            (bar_width, effective, "#bf8700"),
        ):
            height = value / axis_max * plot_height
            body.append(
                f'<rect x="{group_left + offset:.1f}" '
                f'y="{plot_top + plot_height - height:.1f}" width="{bar_width:.1f}" '
                f'height="{height:.1f}" fill="{fill}" rx="1" />'
            )
            body.append(
                svg_text(
                    group_left + offset + bar_width * 0.5,
                    plot_top + plot_height - height - 6,
                    f"{value:,.1f}" if per_standard_month else f"{int(value):,}",
                    css_class="value",
                    anchor="middle",
                )
            )
    body.extend(
        [
            '<rect x="90" y="710" width="18" height="18" fill="#0969da" />',
            svg_text(116, 724, "Source events", css_class="note"),
            '<rect x="245" y="710" width="18" height="18" fill="#bf8700" />',
            svg_text(271, 724, "Source events plus reruns", css_class="note"),
            '<rect x="500" y="710" width="18" height="18" fill="#fff8c5" />',
            svg_text(526, 724, "Partial coverage", css_class="note"),
            '<rect x="680" y="710" width="18" height="18" fill="#ddf4ff" />',
            svg_text(706, 724, "In progress", css_class="note"),
            svg_text(
                90,
                775,
                "Cycles are named for the target release. One month is 30.4375 days;"
                " partial cycles are not complete observations.",
                css_class="note",
            ),
        ]
    )
    path.write_text(
        svg_document(
            (
                "OpenUSD triggering-event rate by release"
                if per_standard_month
                else "OpenUSD triggering events by release"
            ),
            "Source-trigger counts with and without rerun events by release cycle.",
            body,
        ),
        encoding="utf-8",
    )


def render_compute_svg(rows: Sequence[dict], path: Path, *, per_standard_month: bool):
    if per_standard_month:
        series = [
            ("linux_x64_minutes_per_30_4375_days", "#1f883d", "Linux x64"),
            ("macos_minutes_per_30_4375_days", "#8250df", "macOS"),
            ("windows_minutes_per_30_4375_days", "#0969da", "Windows"),
        ]
    else:
        series = [
            ("linux_x64_runner_minutes", "#1f883d", "Linux x64"),
            ("macos_runner_minutes", "#8250df", "macOS"),
            ("windows_runner_minutes", "#0969da", "Windows"),
        ]
    values = [float(row[key]) for row in rows for key, _, _ in series]
    axis_max, axis_step = nice_axis_max(max(values, default=1.0))
    plot_left, plot_top = 90.0, 145.0
    plot_width, plot_height = 1460.0, 480.0
    cell_width = plot_width / len(rows)
    body = [
        svg_text(
            55,
            52,
            (
                "Core Runner-Minute Rate by Release (30.4375 Days)"
                if per_standard_month
                else "Core BuildUSD Runner Minutes by Release"
            ),
            css_class="title",
        ),
        svg_text(
            55,
            82,
            "Linux x64, macOS, and Windows build jobs only; Wasm and packaging are"
            " excluded.",
            css_class="subtitle",
        ),
        *render_coverage_background(rows, plot_left, plot_top, plot_width, plot_height),
        *render_release_axes(
            rows, axis_max, axis_step, plot_left, plot_top, plot_width, plot_height
        ),
    ]
    for index, row in enumerate(rows):
        group_left = plot_left + index * cell_width + cell_width * 0.1
        bar_width = cell_width * 0.8 / len(series)
        for series_index, (key, fill, _) in enumerate(series):
            value = float(row[key])
            height = value / axis_max * plot_height
            y = plot_top + plot_height - height
            body.append(
                f'<rect x="{group_left + series_index * bar_width:.1f}" '
                f'y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{height:.1f}" fill="{fill}" rx="1" />'
            )
            body.append(
                svg_text(
                    group_left + (series_index + 0.5) * bar_width,
                    y - 6,
                    f"{value:,.0f}",
                    css_class="value",
                    anchor="middle",
                )
            )
    legend_x = 90
    for _, fill, label in series:
        x = legend_x
        body.append(f'<rect x="{x}" y="710" width="18" height="18" fill="{fill}" />')
        body.append(svg_text(x + 26, 724, label, css_class="note"))
        legend_x += 125
    body.extend(
        [
            '<rect x="500" y="710" width="18" height="18" fill="#fff8c5" />',
            svg_text(526, 724, "Partial coverage", css_class="note"),
            '<rect x="680" y="710" width="18" height="18" fill="#ddf4ff" />',
            svg_text(706, 724, "In progress", css_class="note"),
        ]
    )
    body.append(
        svg_text(
            90,
            775,
            "Raw jobs remain cached. Summary minutes include only exact BuildUSD jobs:"
            " Linux, macOS, and Windows.",
            css_class="note",
        )
    )
    path.write_text(
        svg_document(
            (
                "Core BuildUSD runner-minute rate by release"
                if per_standard_month
                else "Core BuildUSD runner minutes by release"
            ),
            "Core platform runner minutes by OpenUSD release cycle.",
            body,
        ),
        encoding="utf-8",
    )


def export_release_triggering_events(
    database_dir: Path,
    csv_path: Path,
    svg_path: Path,
    rate_svg_path: Path,
    compute_svg_path: Path,
    compute_rate_svg_path: Path,
):
    rows = release_triggering_event_rows(database_dir)
    fieldnames = tuple(rows[0])
    write_csv(csv_path, fieldnames, rows)
    render_trigger_count_svg(rows, svg_path, per_standard_month=False)
    render_trigger_count_svg(rows, rate_svg_path, per_standard_month=True)
    render_compute_svg(rows, compute_svg_path, per_standard_month=False)
    render_compute_svg(rows, compute_rate_svg_path, per_standard_month=True)
    print(f"Wrote {csv_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {rate_svg_path}")
    print(f"Wrote {compute_svg_path}")
    print(f"Wrote {compute_rate_svg_path}")


###############################################################################
# Forecasting
###############################################################################


def observed_monthly_trigger_rows(database_dir: Path) -> list[dict]:
    data_end = cached_action_end(database_dir)
    end = first_of_month(data_end.date())
    start = COMPLETE_TRIGGER_COVERAGE_START
    if end <= start:
        raise RuntimeError("No complete Actions months are cached")
    run_rows = workflow_run_export_rows(database_dir, start, end)
    job_rows = workflow_job_export_rows(database_dir, start, end)
    source_events = infer_source_events(run_rows)
    rerun_events, _ = infer_rerun_events(run_rows, job_rows)
    source_by_month = collections.Counter(
        event["started_at"].strftime("%Y-%m") for event in source_events
    )
    reruns_by_month = collections.Counter(
        event["started_at"].strftime("%Y-%m") for event in rerun_events
    )
    return [
        {
            "month": month.strftime("%Y-%m"),
            "source_events": source_by_month[month.strftime("%Y-%m")],
            "events_including_reruns": (
                source_by_month[month.strftime("%Y-%m")]
                + reruns_by_month[month.strftime("%Y-%m")]
            ),
        }
        for month in iter_months(start, end)
    ]


def forecast_baseline(release_rows: Sequence[dict]) -> float:
    current_rows = [
        row for row in release_rows if row["coverage_status"] == "in_progress"
    ]
    if len(current_rows) != 1:
        raise RuntimeError("Exactly one in-progress release cycle is required")
    current = current_rows[0]
    return float(current["events_including_reruns_per_30_4375_days"])


def seasonal_forecast_values(
    starting_level: float, release_growth: float, months: int = FORECAST_MONTHS
) -> list[dict]:
    phase_count = len(RELEASE_PHASES)
    if months % phase_count:
        raise ValueError("Forecast length must contain complete release cycles")
    values = []
    for cycle_start in range(0, months, phase_count):
        release_cycle = cycle_start // phase_count + 1
        trend_level = starting_level + release_growth * release_cycle
        raw_values = [trend_level * multiplier for _, multiplier in RELEASE_PHASES]
        scale = trend_level * phase_count / sum(raw_values)
        for offset, (phase, multiplier) in enumerate(RELEASE_PHASES):
            values.append(
                {
                    "forecast_month": cycle_start + offset + 1,
                    "release_cycle": release_cycle,
                    "release_phase": phase,
                    "release_phase_multiplier": multiplier,
                    "trend_level": trend_level,
                    "seasonal_value": raw_values[offset] * scale,
                }
            )
    return values


def least_squares_line(
    xs: Sequence[float], values: Sequence[float]
) -> tuple[float, float]:
    if len(xs) != len(values) or len(xs) < 2:
        raise ValueError("A trend line requires at least two paired values")
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(values)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("A trend line requires distinct x values")
    slope = (
        sum((x - x_mean) * (value - y_mean) for x, value in zip(xs, values))
        / denominator
    )
    intercept = y_mean - slope * x_mean
    return slope, intercept


def observed_release_trend(
    actual_rows: Sequence[dict],
) -> tuple[list[int], list[float], float, float]:
    indexes = [
        index
        for index, row in enumerate(actual_rows)
        if row["coverage_status"] in TREND_COVERAGE_STATUSES
    ]
    values = [
        float(actual_rows[index]["events_including_reruns_per_30_4375_days"])
        for index in indexes
    ]
    slope, intercept = least_squares_line(indexes, values)
    return indexes, values, slope, intercept


def forecast_release_rows(
    actual_rows: Sequence[dict], release_count: int
) -> list[dict]:
    _, _, slope, intercept = observed_release_trend(actual_rows)
    last_actual_index = len(actual_rows) - 1
    last_fitted_value = intercept + slope * last_actual_index
    rows = []
    for offset in range(1, release_count + 1):
        row = {"release_cycle": offset}
        for scenario, multiplier in FORECAST_SCENARIO_GROWTH.items():
            key = f"{scenario}_events_including_reruns"
            value = last_fitted_value + slope * multiplier * offset
            row[key] = value
            row[f"{key}_per_release"] = value * len(RELEASE_PHASES)
        rows.append(row)
    return rows


def render_release_forecast_svg(
    actual_rows: Sequence[dict],
    forecast_rows: Sequence[dict],
    path: Path,
):
    actual_key = "events_including_reruns_per_30_4375_days"
    actual_values = [float(row[actual_key]) for row in actual_rows]
    trend_indexes, _, trend_slope, trend_intercept = observed_release_trend(actual_rows)
    trend_values = [trend_intercept + trend_slope * index for index in trend_indexes]
    forecast_series = {
        scenario: [
            float(row[f"{scenario}_events_including_reruns"]) for row in forecast_rows
        ]
        for scenario in FORECAST_SCENARIO_GROWTH
    }
    all_values = actual_values + [
        value for values in forecast_series.values() for value in values
    ]
    axis_max, axis_step = nice_axis_max(max(all_values))
    plot_left, plot_top = 90.0, 145.0
    plot_width, plot_height = 1460.0, 480.0
    total_points = len(actual_rows) + len(forecast_rows)

    def point(index: float, value: float) -> tuple[float, float]:
        x = plot_left + index * plot_width / (total_points - 1)
        y = plot_top + plot_height - value / axis_max * plot_height
        return x, y

    def path_data(points: Sequence[tuple[float, float]]) -> str:
        return " ".join(
            f"{'M' if index == 0 else 'L'} {x:.1f} {y:.1f}"
            for index, (x, y) in enumerate(points)
        )

    title = "OpenUSD Per-Release Trigger Rate Including Reruns: Actual and Forecast"
    body = [
        svg_text(55, 52, title, css_class="title"),
        svg_text(
            55,
            82,
            "Actual events per 30.4375 days by release, with fitted-trend "
            "low/mid/high five-year scenarios.",
            css_class="subtitle",
        ),
    ]
    tick = 0.0
    while tick <= axis_max + 1e-9:
        y = plot_top + plot_height - tick / axis_max * plot_height
        body.append(
            f'<line x1="{plot_left:.1f}" y1="{y:.1f}" '
            f'x2="{plot_left + plot_width:.1f}" y2="{y:.1f}" class="grid" />'
        )
        body.append(
            svg_text(
                plot_left - 12,
                y + 4,
                f"{tick:,.0f}",
                css_class="axis",
                anchor="end",
            )
        )
        tick += axis_step
    body.append(
        f'<line x1="{plot_left:.1f}" y1="{plot_top + plot_height:.1f}" '
        f'x2="{plot_left + plot_width:.1f}" y2="{plot_top + plot_height:.1f}" '
        'class="axis-line" />'
    )
    forecast_start = len(actual_rows)
    for year in range(1, FORECAST_YEARS + 1):
        boundary_index = forecast_start + year * RELEASES_PER_YEAR - 1
        x, _ = point(boundary_index, 0)
        body.append(
            f'<line x1="{x:.1f}" y1="{plot_top:.1f}" x2="{x:.1f}" '
            f'y2="{plot_top + plot_height:.1f}" class="grid" />'
        )

    actual_points = [point(index, value) for index, value in enumerate(actual_values)]
    body.append(
        f'<path d="{path_data(actual_points)}" fill="none" stroke="#24292f" '
        'stroke-width="3" />'
    )
    coverage_colors = {
        "complete": "#24292f",
        "in_progress": "#0969da",
        "partial": "#bf8700",
    }
    for index, (row, (x, y)) in enumerate(zip(actual_rows, actual_points)):
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" '
            f'fill="{coverage_colors[row["coverage_status"]]}" />'
        )
        label_y = plot_top + plot_height + 26 + (index % 2) * 18
        body.append(
            svg_text(
                x,
                label_y,
                row["release_cycle"],
                css_class="axis",
                anchor="middle",
            )
        )

    trend_points = [
        point(index, value) for index, value in zip(trend_indexes, trend_values)
    ]
    body.append(
        f'<path d="{path_data(trend_points)}" fill="none" stroke="#6e7781" '
        'stroke-width="2" stroke-dasharray="8 5" />'
    )
    divider_x, _ = point(forecast_start - 0.5, 0)
    body.append(
        f'<line x1="{divider_x:.1f}" y1="{plot_top:.1f}" x2="{divider_x:.1f}" '
        f'y2="{plot_top + plot_height:.1f}" stroke="#57606a" stroke-width="2" />'
    )
    body.append(
        svg_text(divider_x + 8, plot_top + 16, "Projected releases", css_class="axis")
    )
    colors = {"low": "#1f883d", "mid": "#0969da", "high": "#bf8700"}
    forecast_line_style = ' stroke-dasharray="2 6" stroke-linecap="round"'
    last_actual_index = len(actual_rows) - 1
    fitted_boundary = trend_intercept + trend_slope * last_actual_index
    for scenario, values in forecast_series.items():
        points = [point(last_actual_index, fitted_boundary)] + [
            point(forecast_start + index, value) for index, value in enumerate(values)
        ]
        body.append(
            f'<path d="{path_data(points)}" fill="none" '
            f'stroke="{colors[scenario]}" stroke-width="3"'
            f"{forecast_line_style} />"
        )
    marker_offsets = {"low": 16, "mid": -10, "high": -10}
    for forecast_year in (1, FORECAST_YEARS):
        release_index = forecast_year * RELEASES_PER_YEAR - 1
        chart_index = forecast_start + release_index
        for scenario, values in forecast_series.items():
            x, y = point(chart_index, values[release_index])
            body.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" '
                f'fill="{colors[scenario]}" stroke="#ffffff" stroke-width="2" />'
            )
            body.append(
                svg_text(
                    x + 8,
                    y + marker_offsets[scenario],
                    f"{values[release_index]:.0f}",
                    css_class="axis",
                )
            )
    for year in range(1, FORECAST_YEARS + 1):
        center_index = forecast_start + (year - 1) * RELEASES_PER_YEAR + 1.5
        x, _ = point(center_index, 0)
        body.append(
            svg_text(
                x,
                plot_top + plot_height + 26,
                f"Forecast year {year}",
                css_class="axis",
                anchor="middle",
            )
        )

    legend = [
        ("#24292f", "Actual", ""),
        ("#6e7781", "Observed trend", ' stroke-dasharray="8 5"'),
        (
            colors["low"],
            f"Low ({FORECAST_SCENARIO_GROWTH['low']:.0%} slope)",
            forecast_line_style,
        ),
        (
            colors["mid"],
            f"Mid ({FORECAST_SCENARIO_GROWTH['mid']:.0%} slope)",
            forecast_line_style,
        ),
        (
            colors["high"],
            f"High ({FORECAST_SCENARIO_GROWTH['high']:.0%} slope)",
            forecast_line_style,
        ),
    ]
    legend_x = 90
    for color, label, dash_attribute in legend:
        body.append(
            f'<line x1="{legend_x}" y1="716" x2="{legend_x + 24}" y2="716" '
            f'stroke="{color}" stroke-width="4"{dash_attribute} />'
        )
        body.append(svg_text(legend_x + 32, 721, label, css_class="note"))
        legend_x += 260
    body.append(
        svg_text(
            90,
            775,
            "Partial actual cycles are gold; the current in-progress cycle is blue. "
            "The trend line uses complete and in-progress cycles.",
            css_class="note",
        )
    )
    path.write_text(
        svg_document(
            title, "Actual and projected per-release monthly trigger rates.", body
        ),
        encoding="utf-8",
    )


def annual_release_forecast_totals(release_rows: Sequence[dict]) -> list[dict]:
    totals = []
    for year_index in range(FORECAST_YEARS):
        year_rows = release_rows[
            year_index * RELEASES_PER_YEAR : (year_index + 1) * RELEASES_PER_YEAR
        ]
        row = {"year": year_index + 1}
        for scenario in FORECAST_SCENARIO_GROWTH:
            key = f"{scenario}_events_including_reruns_per_release"
            row[scenario] = sum(item[key] for item in year_rows)
        totals.append(row)
    return totals


def render_report(
    database_dir: Path,
    report_path: Path,
    forecast_csv_path: Path,
    effective_release_forecast_svg_path: Path,
):
    release_rows = release_triggering_event_rows(database_dir)
    actual_rows = observed_monthly_trigger_rows(database_dir)
    starting_level = forecast_baseline(release_rows)
    _, _, fitted_slope, fitted_intercept = observed_release_trend(release_rows)
    release_forecast_rows = forecast_release_rows(release_rows, FORECAST_RELEASES)
    csv_rows = [
        {
            key: f"{value:.1f}" if isinstance(value, float) else value
            for key, value in row.items()
        }
        for row in release_forecast_rows
    ]
    write_csv(forecast_csv_path, tuple(csv_rows[0]), csv_rows)
    render_release_forecast_svg(
        release_rows,
        release_forecast_rows,
        effective_release_forecast_svg_path,
    )
    annual = annual_release_forecast_totals(release_forecast_rows)
    five_year = {
        scenario: sum(row[scenario] for row in annual)
        for scenario in FORECAST_SCENARIO_GROWTH
    }
    fitted_boundary = fitted_intercept + fitted_slope * (len(release_rows) - 1)
    metadata_rows = query_csv(
        database_dir,
        "SELECT meta_key, meta_value FROM sync_metadata ORDER BY meta_key;",
    )
    metadata = {row["meta_key"]: row["meta_value"] for row in metadata_rows}
    phase_values = seasonal_forecast_values(starting_level, 0.0, 3)
    lines = [
        "# OpenUSD GitHub Actions Trigger Forecast",
        "",
        f"Generated: {datetime.date.today().isoformat()}",
        "",
        "## Forecast basis",
        "",
        "The forecast uses inferred source-trigger events from cached GitHub Actions",
        "runs. Runs sharing a source identity within five minutes count once. Reruns",
        "are clustered separately and included in every forecast value.",
        "",
        f"The current release-cycle rate rounds to **{starting_level:.0f} triggering",
        "events including reruns** per 30.4375 days.",
        "The forecast starts at the least-squares line's value at the current",
        "release boundary. Its linear slopes do not compound exponentially.",
        "",
        "Partial release cycles are excluded from the fit because their event counts",
        "are incomplete.",
        "",
        f"The fitted boundary is **{fitted_boundary:.1f} events/month**, and the",
        f"observed slope is **{fitted_slope:.1f} events/month per release**.",
        "",
        (
            "| Scenario | Observed slope retained | Increase per release | "
            "Five-year events including reruns |"
        ),
        "|---|---:|---:|---:|",
    ]
    for scenario, multiplier in FORECAST_SCENARIO_GROWTH.items():
        lines.append(
            f"| {scenario.title()} | {multiplier:.0%} | "
            f"+{fitted_slope * multiplier:.1f} | "
            f"{five_year[scenario]:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Release-cycle seasonality",
            "",
            "The recent burst is modeled as release-cycle seasonality rather than a",
            "permanent increase in the trend. Each projected release cycle has three",
            "equal 30.4375-day phases. Phase values are normalized within every cycle",
            "so seasonality changes timing and peak capacity, not the scenario total.",
            "",
            "| Release phase | Multiplier | Events/month at the current level |",
            "|---|---:|---:|",
        ]
    )
    for value in phase_values:
        lines.append(
            f"| {value['release_phase'].title()} | "
            f"{value['release_phase_multiplier']:.1f}x | "
            f"{value['seasonal_value']:.0f} |"
        )
    lines.extend(
        [
            "",
            "The cycle average is the cost-planning quantity. The late-cycle value",
            "is the runner-capacity quantity.",
            "",
            "## Annual totals",
            "",
            "| Year | Low incl. reruns | Mid incl. reruns | High incl. reruns |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in annual:
        lines.append(
            f"| Year {row['year']} | "
            f"{row['low']:,.0f} | "
            f"{row['mid']:,.0f} | "
            f"{row['high']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Observed release-cycle rates",
            "",
            "| Release cycle | Coverage | Events including reruns/month |",
            "|---|---|---:|",
        ]
    )
    for row in release_rows:
        lines.append(
            f"| {row['release_cycle']} | {row['coverage_status']} | "
            f"{float(row['events_including_reruns_per_30_4375_days']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Generated forecast artifacts",
            "",
            (
                f"- `{forecast_csv_path.name}`: {FORECAST_RELEASES} projected "
                "release cycles, with monthly rates and three-month totals"
            ),
            (
                f"- `{effective_release_forecast_svg_path.name}`: rates including "
                "reruns by actual and projected release"
            ),
            "",
            "The SVG shows actual per-release monthly rates, a least-squares line over",
            (
                "the complete and in-progress cycles, and "
                f"{FORECAST_RELEASES} projected release-cycle"
            ),
            "averages. Inner-release monthly seasonality is intentionally omitted.",
            "",
            "## Limitations",
            "",
            "- The scenario slopes and phase multipliers are planning assumptions",
            (
                "  grounded in the available Actions observations, not statistically"
                " stable"
            ),
            "  estimates from many release cycles.",
            "- Refit the level and reconsider the slopes after each completed release.",
            "- Trigger forecasts do not predict another workflow's duration. Apply",
            "  platform-specific job times when converting events into runner minutes.",
            "- Workflow configuration and triggering rules can change.",
            "",
            "## Data coverage and reproducibility",
            "",
            f"- Repository ID: `{REPOSITORY_ID}`",
            f"- Repository: `{REPOSITORY}`",
            (
                "- Cached Actions range ends: "
                f"{metadata.get('actions_end_date_exclusive', 'unknown')} (exclusive)"
            ),
            f"- Complete monthly Actions observations: {len(actual_rows)}",
            (
                "- Cached workflows/runs/jobs: "
                f"{metadata.get('actions_workflow_count', 'unknown')}/"
                f"{metadata.get('actions_run_count', 'unknown')}/"
                f"{metadata.get('actions_job_count', 'unknown')}"
            ),
            "",
            "```bash",
            "./run_openusd_usage.sh report",
            "```",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Wrote {forecast_csv_path}")
    print(f"Wrote {effective_release_forecast_svg_path}")


###############################################################################
# CLI
###############################################################################


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=(
            "export-releases",
            "init",
            "report",
            "sync-actions",
            "sync-all",
            "sync-pr-events",
            "sync-pr-timelines",
            "sync-releases",
        ),
    )
    parser.add_argument("--database-dir", type=Path, default=DEFAULT_DATABASE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--forecast-csv", type=Path, default=DEFAULT_FORECAST_CSV)
    parser.add_argument(
        "--effective-release-forecast-svg",
        type=Path,
        default=DEFAULT_EFFECTIVE_RELEASE_FORECAST_SVG,
    )
    parser.add_argument("--release-csv", type=Path, default=DEFAULT_RELEASE_CSV)
    parser.add_argument("--release-svg", type=Path, default=DEFAULT_RELEASE_SVG)
    parser.add_argument(
        "--release-rate-svg", type=Path, default=DEFAULT_RELEASE_RATE_SVG
    )
    parser.add_argument("--compute-svg", type=Path, default=DEFAULT_COMPUTE_SVG)
    parser.add_argument(
        "--compute-rate-svg", type=Path, default=DEFAULT_COMPUTE_RATE_SVG
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=five_years_before(datetime.date.today()),
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=datetime.date.today() + datetime.timedelta(days=1),
        help="Exclusive upper bound",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)
    try:
        database_dir = args.database_dir.resolve()
        initialize_database(database_dir)
        stage_dir = database_dir.parent / "openusd_usage_stage"
        if args.command in ("sync-all", "sync-pr-events"):
            sync_pull_request_events(
                database_dir, stage_dir, args.start_date, args.end_date
            )
        if args.command in ("sync-all", "sync-pr-timelines"):
            sync_pull_request_timelines(database_dir, stage_dir)
        if args.command in ("sync-actions", "sync-all"):
            sync_actions(database_dir, stage_dir, args.start_date, args.end_date)
        if args.command in ("sync-releases", "sync-all"):
            sync_releases(database_dir, stage_dir)
        if args.command in ("report", "sync-all"):
            render_report(
                database_dir,
                args.report.resolve(),
                args.forecast_csv.resolve(),
                args.effective_release_forecast_svg.resolve(),
            )
        if args.command == "export-releases":
            export_release_triggering_events(
                database_dir,
                args.release_csv.resolve(),
                args.release_svg.resolve(),
                args.release_rate_svg.resolve(),
                args.compute_svg.resolve(),
                args.compute_rate_svg.resolve(),
            )
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
