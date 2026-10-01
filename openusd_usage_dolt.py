#!/usr/bin/env python

"""Cache OpenUSD pull-request and GitHub Actions usage in Dolt and forecast demand.

The pull-request history comes from ClickHouse's public GitHub Archive mirror.
GitHub Actions workflows, runs, jobs, and steps come from the GitHub REST API
through the authenticated ``gh`` CLI.

Examples:
    ./run_openusd_usage.sh sync-all --start-date 2021-09-16
    ./run_openusd_usage.sh export-monthly
    ./run_openusd_usage.sh report
    ./run_openusd_usage.sh sync-pr-events --start-date 2021-09-16
"""

import argparse
import csv
import datetime
import hashlib
import html
import json
import math
import statistics
import subprocess
import sys
import traceback

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests

###############################################################################
# Constants
###############################################################################


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE_DIR = THIS_DIR / ".cache" / "openusd_usage_dolt"
DEFAULT_REPORT = THIS_DIR / "openusd_usage_forecast.md"
DEFAULT_MONTHLY_CSV = THIS_DIR / "openusd_triggering_events_monthly.csv"
DEFAULT_MONTHLY_SVG = THIS_DIR / "openusd_triggering_events_monthly.svg"
DEFAULT_COMPUTE_SVG = THIS_DIR / "openusd_runner_minutes_monthly.svg"
REPOSITORY = "PixarAnimationStudios/OpenUSD"
REPOSITORY_ID = 58168143
BUILD_WORKFLOW_NAME = "BuildUSD"
CLICKHOUSE_URL = "https://sql-clickhouse.clickhouse.com/"
CLICKHOUSE_USER = "demo"
FORECAST_MONTHS = 60
TRIGGER_ACTIONS = ("opened", "synchronize", "reopened")
# Earlier cached months contain BuildUSD records, but complete all-workflow
# collection starts here. Treat earlier months as unavailable or partial.
COMPLETE_TRIGGER_COVERAGE_START = datetime.date(2025, 9, 1)
TRIGGER_CLUSTER_MINUTES = 5

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sync_metadata (
    meta_key VARCHAR(255) PRIMARY KEY,
    meta_value LONGTEXT NOT NULL,
    updated_at DATETIME(6) NOT NULL
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


@dataclass(frozen=True)
class ModelResult:
    name: str
    mae: float
    rmse: float
    fit: Callable[[float], float]


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


def shift_months(day: datetime.date, offset: int) -> datetime.date:
    month_index = day.year * 12 + day.month - 1 + offset
    year, month = divmod(month_index, 12)
    return datetime.date(year, month + 1, 1)


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
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
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
        "SELECT j.job_id, j.run_id, j.started_at, j.completed_at, "
        "j.labels_json, j.conclusion, j.raw_json FROM workflow_jobs j "
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


def runner_family(labels_json: str) -> str:
    labels = " ".join(json.loads(labels_json)).lower()
    if "macos" in labels:
        return "macos"
    if "windows" in labels:
        return "windows"
    if any(label in labels for label in ("linux", "ubuntu")):
        return "linux"
    return "other"


def job_minutes(job: dict[str, str]) -> float:
    if not job["started_at"] or not job["completed_at"]:
        return 0.0
    started_at = parse_database_datetime(job["started_at"])
    completed_at = parse_database_datetime(job["completed_at"])
    return max(0.0, (completed_at - started_at).total_seconds() / 60)


def monthly_triggering_event_rows(
    database_dir: Path,
    start: datetime.date,
    end: datetime.date,
) -> list[dict]:
    run_rows = workflow_run_export_rows(database_dir, start, end)
    job_rows = workflow_job_export_rows(database_dir, start, end)
    source_events = infer_source_events(run_rows)
    rerun_events, rerun_attempts = infer_rerun_events(run_rows, job_rows)

    source_by_month: dict[str, int] = {}
    reruns_by_month: dict[str, int] = {}
    attempts_by_month: dict[str, int] = {}
    runs_by_month: dict[str, int] = {}
    minutes_by_month: dict[str, dict[str, float]] = {}
    for event in source_events:
        month = event["started_at"].strftime("%Y-%m")
        source_by_month[month] = source_by_month.get(month, 0) + 1
    for event in rerun_events:
        month = event["started_at"].strftime("%Y-%m")
        reruns_by_month[month] = reruns_by_month.get(month, 0) + 1
    for attempt in rerun_attempts:
        month = attempt["started_at"].strftime("%Y-%m")
        attempts_by_month[month] = attempts_by_month.get(month, 0) + 1
    for run in run_rows:
        month = run["created_at"][:7]
        runs_by_month[month] = runs_by_month.get(month, 0) + 1
    for job in job_rows:
        if not job["started_at"]:
            continue
        month = job["started_at"][:7]
        family = runner_family(job["labels_json"])
        monthly_minutes = minutes_by_month.setdefault(
            month,
            {"linux": 0.0, "macos": 0.0, "other": 0.0, "windows": 0.0},
        )
        monthly_minutes[family] += job_minutes(job)

    rows = []
    first_observed = min(
        (first_of_month(event["started_at"].date()) for event in source_events),
        default=end,
    )
    for month in iter_months(start, end):
        month_key = month.strftime("%Y-%m")
        if month < first_observed:
            rows.append(
                {
                    "month": month_key,
                    "coverage_status": "unavailable",
                    "workflow_data_source": "not_available",
                    "source_triggering_events": "",
                    "rerun_triggering_events": "",
                    "triggering_events_including_reruns": "",
                    "workflow_runs": "",
                    "workflow_rerun_attempts": "",
                    "linux_runner_minutes": "",
                    "macos_runner_minutes": "",
                    "windows_runner_minutes": "",
                    "other_runner_minutes": "",
                    "linux_minutes_per_source_event": "",
                    "macos_minutes_per_source_event": "",
                    "windows_minutes_per_source_event": "",
                    "other_minutes_per_source_event": "",
                }
            )
            continue
        coverage = "complete" if month >= COMPLETE_TRIGGER_COVERAGE_START else "partial"
        source_count = source_by_month.get(month_key, 0)
        rerun_count = reruns_by_month.get(month_key, 0)
        effective_count = source_count + rerun_count
        minutes = minutes_by_month.get(
            month_key,
            {"linux": 0.0, "macos": 0.0, "other": 0.0, "windows": 0.0},
        )
        rows.append(
            {
                "month": month_key,
                "coverage_status": coverage,
                "workflow_data_source": "github_rest_api_cached_in_dolt",
                "source_triggering_events": source_count,
                "rerun_triggering_events": rerun_count,
                "triggering_events_including_reruns": effective_count,
                "workflow_runs": runs_by_month.get(month_key, 0),
                "workflow_rerun_attempts": attempts_by_month.get(month_key, 0),
                "linux_runner_minutes": f"{minutes['linux']:.1f}",
                "macos_runner_minutes": f"{minutes['macos']:.1f}",
                "windows_runner_minutes": f"{minutes['windows']:.1f}",
                "other_runner_minutes": f"{minutes['other']:.1f}",
                "linux_minutes_per_source_event": (
                    f"{minutes['linux'] / source_count:.1f}" if source_count else ""
                ),
                "macos_minutes_per_source_event": (
                    f"{minutes['macos'] / source_count:.1f}" if source_count else ""
                ),
                "windows_minutes_per_source_event": (
                    f"{minutes['windows'] / source_count:.1f}" if source_count else ""
                ),
                "other_minutes_per_source_event": (
                    f"{minutes['other'] / source_count:.1f}" if source_count else ""
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
                '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="820" '
                'viewBox="0 0 1600 820" role="img">'
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
        fill = "#f6f8fa" if row["coverage_status"] == "unavailable" else "#fff8c5"
        output.append(
            f'<rect x="{plot_left + index * cell_width:.1f}" y="{plot_top:.1f}" '
            f'width="{cell_width + 0.2:.1f}" height="{plot_height:.1f}" '
            f'fill="{fill}" />'
        )
    return output


def render_axes(
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
        month_number = int(row["month"][5:7])
        if month_number not in (1, 4, 7, 10):
            continue
        x = plot_left + (index + 0.5) * cell_width
        output.append(
            svg_text(
                x,
                plot_top + plot_height + 24,
                row["month"],
                css_class="axis",
                anchor="end",
                transform=f"rotate(-45 {x:.1f} {plot_top + plot_height + 24:.1f})",
            )
        )
    return output


def render_trigger_count_svg(rows: Sequence[dict], path: Path):
    values = [
        float(row["triggering_events_including_reruns"])
        for row in rows
        if row["triggering_events_including_reruns"] != ""
    ]
    axis_max, axis_step = nice_axis_max(max(values, default=1))
    plot_left, plot_top = 90.0, 145.0
    plot_width, plot_height = 1460.0, 480.0
    cell_width = plot_width / len(rows)
    body = [
        svg_text(55, 52, "OpenUSD Triggering Events per Month", css_class="title"),
        svg_text(
            55,
            82,
            "One source event counts once across workflows; the second bar adds"
            " inferred rerun events.",
            css_class="subtitle",
        ),
        *render_coverage_background(rows, plot_left, plot_top, plot_width, plot_height),
        *render_axes(
            rows, axis_max, axis_step, plot_left, plot_top, plot_width, plot_height
        ),
    ]
    for index, row in enumerate(rows):
        if row["source_triggering_events"] == "":
            continue
        source = float(row["source_triggering_events"])
        effective = float(row["triggering_events_including_reruns"])
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
        if effective:
            body.append(
                svg_text(
                    group_left + bar_width * 1.5,
                    plot_top + plot_height - effective / axis_max * plot_height - 6,
                    f"{int(effective)}",
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
            '<rect x="680" y="710" width="18" height="18" fill="#f6f8fa" />',
            svg_text(706, 724, "Unavailable", css_class="note"),
            svg_text(
                90,
                775,
                "Inference clusters runs sharing a commit identity within five minutes."
                " Reruns use job attempt start times.",
                css_class="note",
            ),
        ]
    )
    path.write_text(
        svg_document(
            "OpenUSD triggering events per month",
            "Monthly source-trigger counts with and without rerun events.",
            body,
        ),
        encoding="utf-8",
    )


def render_compute_svg(rows: Sequence[dict], path: Path):
    series = [
        ("linux_runner_minutes", "#1f883d", "Linux"),
        ("windows_runner_minutes", "#0969da", "Windows"),
        ("macos_runner_minutes", "#8250df", "macOS"),
    ]
    if any(row["other_runner_minutes"] not in ("", "0.0") for row in rows):
        series.append(("other_runner_minutes", "#bf8700", "Other"))
    values = [
        float(row[key])
        for row in rows
        if row["linux_runner_minutes"] != ""
        for key, _, _ in series
    ]
    axis_max, axis_step = nice_axis_max(max(values, default=1.0))
    plot_left, plot_top = 90.0, 145.0
    plot_width, plot_height = 1460.0, 480.0
    cell_width = plot_width / len(rows)
    body = [
        svg_text(
            55,
            52,
            "Observed OpenUSD Runner Minutes by Platform",
            css_class="title",
        ),
        svg_text(
            55,
            82,
            "Context only: OpenUSD workflow duration is not assumed to predict another"
            " workflow.",
            css_class="subtitle",
        ),
        *render_coverage_background(rows, plot_left, plot_top, plot_width, plot_height),
        *render_axes(
            rows, axis_max, axis_step, plot_left, plot_top, plot_width, plot_height
        ),
    ]
    for index, row in enumerate(rows):
        if row["linux_runner_minutes"] == "":
            continue
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
    legend_x = 90
    for _, fill, label in series:
        x = legend_x
        body.append(f'<rect x="{x}" y="710" width="18" height="18" fill="{fill}" />')
        body.append(svg_text(x + 26, 724, label, css_class="note"))
        legend_x += 125
    body.append(
        svg_text(
            90,
            775,
            "Platform series are not combined. Durations include rerun attempts;"
            " skipped jobs contribute zero.",
            css_class="note",
        )
    )
    path.write_text(
        svg_document(
            "Observed OpenUSD runner minutes by platform",
            "Monthly per-platform runner-minutes including workflow rerun attempts.",
            body,
        ),
        encoding="utf-8",
    )


def export_monthly_triggering_events(
    database_dir: Path, csv_path: Path, svg_path: Path, compute_svg_path: Path
):
    end = first_of_month(datetime.date.today())
    start = shift_months(end, -60)
    rows = monthly_triggering_event_rows(database_dir, start, end)
    fieldnames = tuple(rows[0])
    write_csv(csv_path, fieldnames, rows)
    render_trigger_count_svg(rows, svg_path)
    render_compute_svg(rows, compute_svg_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {compute_svg_path}")


###############################################################################
# Forecasting
###############################################################################


def constant_fit(values: Sequence[float]) -> Callable[[float], float]:
    mean = statistics.fmean(values)
    return lambda _: mean


def trailing_mean_fit(values: Sequence[float]) -> Callable[[float], float]:
    mean = statistics.fmean(values[-24:])
    return lambda _: mean


def linear_coefficients(
    xs: Sequence[float], ys: Sequence[float]
) -> tuple[float, float]:
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return y_mean, 0.0
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    return y_mean - slope * x_mean, slope


def linear_fit(values: Sequence[float]) -> Callable[[float], float]:
    intercept, slope = linear_coefficients(list(range(len(values))), values)
    return lambda x: max(0.0, intercept + slope * x)


def exponential_fit(values: Sequence[float]) -> Callable[[float], float]:
    transformed = [math.log1p(max(0.0, value)) for value in values]
    intercept, slope = linear_coefficients(list(range(len(values))), transformed)
    return lambda x: max(0.0, math.exp(intercept + slope * x) - 1.0)


def solve_three_by_three(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Singular quadratic regression matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][3] for row in range(3)]


def quadratic_fit(values: Sequence[float]) -> Callable[[float], float]:
    xs = list(range(len(values)))
    matrix = [
        [len(xs), sum(xs), sum(x**2 for x in xs)],
        [sum(xs), sum(x**2 for x in xs), sum(x**3 for x in xs)],
        [sum(x**2 for x in xs), sum(x**3 for x in xs), sum(x**4 for x in xs)],
    ]
    vector = [
        sum(values),
        sum(x * y for x, y in zip(xs, values)),
        sum((x**2) * y for x, y in zip(xs, values)),
    ]
    intercept, linear, quadratic = solve_three_by_three(matrix, vector)
    return lambda x: max(0.0, intercept + linear * x + quadratic * x**2)


MODEL_FACTORIES: dict[str, Callable[[Sequence[float]], Callable[[float], float]]] = {
    "constant": constant_fit,
    "exponential": exponential_fit,
    "linear": linear_fit,
    "quadratic": quadratic_fit,
    "trailing-24-month mean": trailing_mean_fit,
}


def evaluate_models(values: Sequence[float]) -> list[ModelResult]:
    minimum_training = max(24, len(values) - 24)
    results = []
    for name, factory in MODEL_FACTORIES.items():
        errors = []
        squared_errors = []
        for index in range(minimum_training, len(values)):
            predictor = factory(values[:index])
            error = predictor(index) - values[index]
            errors.append(abs(error))
            squared_errors.append(error**2)
        if not errors:
            raise ValueError("Not enough complete months for model evaluation")
        results.append(
            ModelResult(
                name=name,
                mae=statistics.fmean(errors),
                rmse=math.sqrt(statistics.fmean(squared_errors)),
                fit=factory(values),
            )
        )
    return sorted(results, key=lambda result: (result.mae, result.rmse))


def complete_month_bounds(database_dir: Path) -> tuple[datetime.date, datetime.date]:
    rows = query_csv(
        database_dir,
        "SELECT MIN(event_at) AS first_event, MAX(event_at) AS last_event FROM ("
        "SELECT created_at AS event_at FROM pull_request_events UNION ALL "
        "SELECT event_at FROM pull_request_timeline_events WHERE event_at IS NOT NULL"
        ") events;",
    )
    if not rows or not rows[0]["first_event"]:
        raise RuntimeError("No pull-request events are cached")
    first = datetime.date.fromisoformat(rows[0]["first_event"][:10])
    last = datetime.date.fromisoformat(rows[0]["last_event"][:10])
    metadata = query_csv(
        database_dir,
        "SELECT meta_value FROM sync_metadata WHERE meta_key = 'pr_events_start_date';",
    )
    if metadata:
        first = max(first, datetime.date.fromisoformat(metadata[0]["meta_value"]))
    start = next_month(first_of_month(first)) if first.day != 1 else first
    end = first_of_month(last)
    return start, end


def monthly_trigger_counts(
    database_dir: Path, start: datetime.date, end: datetime.date
) -> list[tuple[datetime.date, float]]:
    actions = ",".join(f"'{action}'" for action in TRIGGER_ACTIONS)
    rows = query_csv(
        database_dir,
        "SELECT DATE_FORMAT(created_at, '%Y-%m-01') AS month, "
        "COUNT(*) AS triggers FROM pull_request_events "
        f"WHERE action IN ({actions}) "
        f"AND created_at >= '{start}' AND created_at < '{end}' "
        "GROUP BY month ORDER BY month;",
    )
    by_month = {row["month"][:10]: float(row["triggers"]) for row in rows}
    return [
        (month, by_month.get(month.isoformat(), 0.0))
        for month in iter_months(start, end)
    ]


def actual_workflow_counts(database_dir: Path) -> dict[str, int]:
    rows = query_csv(
        database_dir,
        "SELECT DATE_FORMAT(r.created_at, '%Y-%m-01') AS month, COUNT(*) AS runs "
        "FROM workflow_runs r JOIN workflows w ON w.workflow_id = r.workflow_id "
        f"WHERE w.name = '{BUILD_WORKFLOW_NAME}' AND r.event = 'pull_request' "
        "GROUP BY month ORDER BY month;",
    )
    return {row["month"][:10]: int(row["runs"]) for row in rows}


def runner_usage_by_label(
    database_dir: Path, end: datetime.date
) -> list[dict[str, str]]:
    trailing_start = end.replace(year=end.year - 1)
    return query_csv(
        database_dir,
        "SELECT j.labels_json AS runner_label, COUNT(*) AS jobs, "
        "ROUND(SUM(CASE WHEN j.completed_at > j.started_at THEN "
        "TIMESTAMPDIFF(SECOND, j.started_at, j.completed_at) ELSE 0 END) / 60, 1) "
        "AS cached_minutes, ROUND(SUM(CASE WHEN "
        f"r.created_at >= '{trailing_start}' AND r.created_at < '{end}' "
        "AND j.completed_at > j.started_at THEN "
        "TIMESTAMPDIFF(SECOND, j.started_at, j.completed_at) ELSE 0 END) / 60, 1) "
        "AS trailing_minutes FROM workflow_jobs j JOIN workflow_runs r "
        "ON r.run_id = j.run_id GROUP BY j.labels_json ORDER BY j.labels_json;",
    )


def yearly_forecast(
    predictor: Callable[[float], float], observed_months: int
) -> list[float]:
    monthly = [predictor(observed_months + index) for index in range(FORECAST_MONTHS)]
    return [sum(monthly[index : index + 12]) for index in range(0, 60, 12)]


def is_plausible_long_horizon(result: ModelResult, values: Sequence[float]) -> bool:
    forecast = yearly_forecast(result.fit, len(values))
    recent_annual = statistics.fmean(values[-24:]) * 12
    return (
        forecast[-1] >= max(1.0, recent_annual * 0.25)
        and forecast[-1] <= recent_annual * 4
    )


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return numerator / (x_scale * y_scale) if x_scale and y_scale else 0.0


def render_report(database_dir: Path, report_path: Path):
    start, end = complete_month_bounds(database_dir)
    monthly = monthly_trigger_counts(database_dir, start, end)
    values = [value for _, value in monthly]
    results = evaluate_models(values)
    plausible_results = [
        result for result in results if is_plausible_long_horizon(result, values)
    ]
    if not plausible_results:
        raise RuntimeError("No forecast model passed the long-horizon guardrail")
    selected = plausible_results[0]
    forecast = yearly_forecast(selected.fit, len(values))
    actual_runs = actual_workflow_counts(database_dir)
    runner_usage = runner_usage_by_label(database_dir, end)
    overlap = [
        (month, int(value), actual_runs[month.isoformat()])
        for month, value in monthly
        if month.isoformat() in actual_runs
    ]
    overlap_proxy = sum(item[1] for item in overlap)
    overlap_actual = sum(item[2] for item in overlap)
    calibration = overlap_actual / overlap_proxy if overlap_proxy else 1.0
    proxy_predictions = [item[1] * calibration for item in overlap]
    actual_values = [item[2] for item in overlap]
    proxy_mae = statistics.fmean(
        abs(predicted - actual)
        for predicted, actual in zip(proxy_predictions, actual_values)
    )
    proxy_correlation = correlation(
        [float(item[1]) for item in overlap],
        [float(item[2]) for item in overlap],
    )
    complete_actuals = [
        count for month, count in sorted(actual_runs.items()) if month < end.isoformat()
    ]
    actual_baseline = statistics.fmean(complete_actuals[-12:]) * 12

    metadata_rows = query_csv(
        database_dir,
        "SELECT meta_key, meta_value FROM sync_metadata ORDER BY meta_key;",
    )
    metadata = {row["meta_key"]: row["meta_value"] for row in metadata_rows}
    actual_months = sorted(actual_runs)
    actual_start = actual_months[0] if actual_months else "unknown"
    actual_end = actual_months[-1] if actual_months else "unknown"
    lines = [
        "# OpenUSD Pull-Request Activity and GitHub Actions Forecast",
        "",
        f"Generated: {datetime.date.today().isoformat()}",
        "",
        "## Data coverage",
        "",
        f"- Repository ID: `{REPOSITORY_ID}`",
        f"- Historical names: `PixarAnimationStudios/USD` and `{REPOSITORY}`",
        f"- Pull-request event source: {metadata.get('pr_events_source', 'unknown')}",
        (
            "- Cached PR-event range:"
            f" {metadata.get('pr_events_start_date', 'unknown')} through"
            f" {metadata.get('pr_events_end_date_exclusive', 'unknown')} (exclusive)"
        ),
        (
            f"- Complete months modeled: {start} through"
            f" {end - datetime.timedelta(days=1)}"
        ),
        f"- Complete monthly observations: {len(values)}",
        "- `BuildUSD` workflow record created: 2024-09-09",
        f"- Cached `BuildUSD` run months: {actual_start} through {actual_end}",
        (
            "- Cached workflows/runs/jobs: "
            f"{metadata.get('actions_workflow_count', 'unknown')}/"
            f"{metadata.get('actions_run_count', 'unknown')}/"
            f"{metadata.get('actions_job_count', 'unknown')}"
        ),
        (
            "- Cached PR timelines/events: "
            f"{metadata.get('pr_timeline_pr_count', 'unknown')}/"
            f"{metadata.get('pr_timeline_event_count', 'unknown')}"
        ),
        "",
        "The stable five-year demand proxy counts PR openings and reopenings. GitHub",
        "Actions also uses `synchronize` as a default `pull_request` activity, but",
        "GitHub's historical APIs do not expose exact push timestamps for that event.",
        "The cached timeline commits use author/committer dates and are therefore not",
        "treated as exact workflow triggers. Calibration against actual runs absorbs",
        "the average effect of synchronizations and reruns during the overlap period.",
        "",
        "## Model comparison",
        "",
        "Models were compared over the latest 24 observations using expanding-window,",
        "one-month-ahead validation. A long-horizon guardrail rejects a model when its",
        "Year 5 forecast is below 25% or above 400% of the recent annualized mean.",
        "The lowest-MAE model that passes the guardrail is selected.",
        "",
        "| Model | Validation MAE | Validation RMSE | Guardrail |",
        "|---|---:|---:|---|",
    ]
    for result in results:
        guardrail = "pass" if result in plausible_results else "reject"
        lines.append(
            f"| {result.name} | {result.mae:.2f} | {result.rmse:.2f} | {guardrail} |"
        )
    lines.extend(
        [
            "",
            f"Selected model: **{selected.name}**.",
            "",
            "## Five-year trigger forecast",
            "",
            (
                "| Forecast year | PR-trigger proxy | Calibrated proxy runs | "
                "Flat actual-run baseline |"
            ),
            "|---|---:|---:|---:|",
        ]
    )
    for year, value in enumerate(forecast, 1):
        lines.append(
            f"| Year {year} | {value:,.0f} | {value * calibration:,.0f} | "
            f"{actual_baseline:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Runner-minute baseline",
            "",
            "The most defensible near-term capacity baseline is the measured runner",
            "time from the latest 12 complete months. The five-year column holds that",
            "annual workload flat; it includes both PR and push workflow runs.",
            "",
            (
                "| Runner label | Cached jobs | Cached minutes | Latest 12-month"
                " minutes | Five-year flat minutes |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for usage in runner_usage:
        trailing_minutes = float(usage["trailing_minutes"])
        lines.append(
            f"| `{usage['runner_label']}` | {int(usage['jobs']):,} | "
            f"{float(usage['cached_minutes']):,.0f} | {trailing_minutes:,.0f} | "
            f"{trailing_minutes * 5:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Proxy validation against actual BuildUSD runs",
            "",
            f"Across {len(overlap)} overlapping complete months, the proxy counted",
            f"{overlap_proxy:,} trigger events and GitHub recorded {overlap_actual:,}",
            f"pull-request workflow runs. The resulting calibration factor is",
            f"**{calibration:.3f} actual runs per proxy trigger**.",
            "",
            f"The monthly proxy-to-run correlation is **{proxy_correlation:.3f}**,",
            f"and its calibrated monthly MAE is **{proxy_mae:.1f} runs**. A weak or",
            "negative correlation means the calibrated proxy is not suitable as a",
            "standalone point estimate. The flat baseline annualizes the latest 12",
            (
                f"complete months of actual workflow data to **{actual_baseline:,.0f}"
                " runs**."
            ),
            "",
            "| Month | Trigger proxy | Actual PR workflow runs |",
            "|---|---:|---:|",
        ]
    )
    for month, proxy, actual in overlap:
        lines.append(f"| {month:%Y-%m} | {proxy} | {actual} |")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Pull-request activity predicts workflow starts, not runner duration.",
            "- Openings/reopenings are a coarse proxy for synchronization-heavy PRs;",
            "  inspect the monthly validation table before using the point forecast.",
            (
                "- Timeline commit events do not preserve exact push times or how"
                " commits were"
            ),
            (
                "  grouped into pushes, so they are cached but excluded from trigger"
                " counts."
            ),
            "- The workflow excludes changes limited to `.github/workflows/**`.",
            "- Workflow configuration, job count, and runner speed can change.",
            "- Five-year extrapolation is substantially more uncertain than the",
            "  one-month validation used to select the model.",
            "- Cost forecasts should multiply predicted runs by measured job duration",
            "  from the cached Actions jobs, not by an assumed one-hour runtime.",
            "",
            "## Database and reproducibility",
            "",
            "The embedded Dolt database is `.cache/openusd_usage_dolt`. It is ignored",
            "by Git, while its internal Dolt commits preserve each completed import",
            "as a queryable data snapshot. Python dependencies are locked with `uv`.",
            "No Dolt SQL server is required or started.",
            "",
            "```bash",
            "./run_openusd_usage.sh sync-all --start-date 2021-09-16",
            "./run_openusd_usage.sh report",
            "dolt -C .cache/openusd_usage_dolt log --oneline",
            "```",
            "",
            "Dolt 2.3 supports Git-backed database remotes. This database is pushed",
            "to the existing GitHub repository through its separate `refs/dolt/data`",
            "ref, so the database history coexists with and does not alter Git",
            "`main`. The configured remote is:",
            "",
            "```text",
            (
                "git+ssh://git@github.com/./pmolodo/"
                "USD-performance-metrics-github-runners-notes.git"
            ),
            "```",
            "",
            "After a successful sync, publish the new Dolt commits with:",
            "",
            "```bash",
            "dolt -C .cache/openusd_usage_dolt push",
            "```",
            "",
            "To reconstruct the cache in a fresh source checkout:",
            "",
            "```bash",
            (
                "dolt clone git@github.com:pmolodo/"
                "USD-performance-metrics-github-runners-notes.git "
                ".cache/openusd_usage_dolt"
            ),
            "```",
            "",
            "See Dolt's [Git remote URL implementation][dolt-git-remote] and",
            "[Git remote integration tests][dolt-git-remote-tests].",
            "",
            (
                "[dolt-git-remote]: "
                "https://github.com/dolthub/dolt/blob/main/go/libraries/"
                "doltcore/env/git_remote_url.go"
            ),
            (
                "[dolt-git-remote-tests]: "
                "https://github.com/dolthub/dolt/blob/main/integration-tests/"
                "bats/remotes-git.bats"
            ),
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


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
            "init",
            "report",
            "export-monthly",
            "sync-actions",
            "sync-all",
            "sync-pr-events",
            "sync-pr-timelines",
        ),
    )
    parser.add_argument("--database-dir", type=Path, default=DEFAULT_DATABASE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--monthly-csv", type=Path, default=DEFAULT_MONTHLY_CSV)
    parser.add_argument("--monthly-svg", type=Path, default=DEFAULT_MONTHLY_SVG)
    parser.add_argument("--compute-svg", type=Path, default=DEFAULT_COMPUTE_SVG)
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
        if args.command in ("report", "sync-all"):
            render_report(database_dir, args.report.resolve())
        if args.command == "export-monthly":
            export_monthly_triggering_events(
                database_dir,
                args.monthly_csv.resolve(),
                args.monthly_svg.resolve(),
                args.compute_svg.resolve(),
            )
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
