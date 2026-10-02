#!/usr/bin/env python

"""Inspect cached Windows timing samples and download public GitHub job logs."""

import argparse
import csv
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
REPO = "PixarAnimationStudios/OpenUSD"


def investigate(output):
    with (ROOT / "openusd_build_timings_jobs.csv").open(newline="") as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if r["platform"] == "windows" and r["filter_result"] == "included"
        ]
    short = [r for r in rows if float(r["job_minutes"]) < 15]
    long = [r for r in rows if float(r["job_minutes"]) > 90]
    print("Successful Windows:", len(rows), "under 15 min:", len(short))
    for month in sorted({r["month"] for r in rows}):
        current = [r for r in rows if r["month"] == month]
        print(
            month,
            "total",
            len(current),
            "under 15",
            sum(float(r["job_minutes"]) < 15 for r in current),
        )
    samples = [
        short[-1],
        min(short, key=lambda r: float(r["job_minutes"])),
        short[0],
        long[-1],
    ]
    for month in ("2026-07", "2026-08", "2026-09"):
        candidates = [r for r in short if r["month"] == month]
        samples.append(
            min(candidates, key=lambda r: abs(float(r["job_minutes"]) - 8.79))
        )
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["Accept"] = "application/vnd.github+json"
    token = subprocess.run(
        ["gh", "auth", "token"], check=True, capture_output=True, text=True
    ).stdout.strip()
    session.headers["Authorization"] = f"Bearer {token}"
    artifact_summary = []
    for row in samples:
        print(
            "\nSAMPLE", row["job_id"], row["started_at"], row["job_minutes"], row["url"]
        )
        job_url = f"https://api.github.com/repos/{REPO}/actions/jobs/{row['job_id']}"
        job_path = output / f"{row['job_id']}.json"
        if not job_path.exists():
            response = session.get(job_url, timeout=60)
            response.raise_for_status()
            job_path.write_text(json.dumps(response.json(), indent=2))
        job = json.loads(job_path.read_text())
        artifact_path = output / f"{row['run_id']}-artifacts.json"
        if not artifact_path.exists():
            artifacts = []
            url = f"https://api.github.com/repos/{REPO}/actions/runs/{row['run_id']}/artifacts?per_page=100"
            while url:
                response = session.get(url, timeout=60)
                response.raise_for_status()
                artifacts.extend(response.json()["artifacts"])
                url = response.links.get("next", {}).get("url")
            artifact_path.write_text(json.dumps(artifacts, indent=2))
        artifacts = json.loads(artifact_path.read_text())
        windows = [a for a in artifacts if a["name"] == "usd-win64"]
        byte_count = sum(a["size_in_bytes"] for a in windows)
        artifact_row = {
            "job_id": row["job_id"],
            "run_id": row["run_id"],
            "started_at": row["started_at"],
            "job_minutes": row["job_minutes"],
            "artifact_name": "usd-win64",
            "matching_artifacts": len(windows),
            "archive_bytes": byte_count if windows else "",
            "archive_mib": byte_count / 1024**2 if windows else "",
            "artifact_created_at": ";".join(a["created_at"] for a in windows),
            "artifact_expired": ";".join(str(a["expired"]) for a in windows),
        }
        artifact_summary.append(artifact_row)
        print("ARTIFACT", artifact_row)
        print(
            "live status", job["status"], job["conclusion"], "sha", job.get("head_sha")
        )
        print(
            "steps",
            [
                (s["name"], s["conclusion"], s.get("started_at"), s.get("completed_at"))
                for s in job["steps"]
                if s["name"] in ("Build USD", "Test USD", "Test USD (flaky)")
            ],
        )
        log_path = output / f"{row['job_id']}.log"
        if not log_path.exists():
            response = session.get(job_url + "/logs", timeout=60)
            if response.status_code in (404, 410):
                print("Log unavailable:", response.status_code)
                continue
            response.raise_for_status()
            log_path.write_text(response.text)
        log = log_path.read_text()
        pattern = re.compile(
            r"##\[error\]|fatal error|tests passed|cache restored|cacheable"
            r" calls|\sHits:|\sMisses:",
            re.I,
        )
        matches = [line for line in log.splitlines() if pattern.search(line)]
        print("LOG EVIDENCE", "\n".join(matches[-55:]))
        sha = job.get("head_sha")
        workflow_path = output / f"{sha}-buildusd.yml"
        if sha and not workflow_path.exists():
            response = requests.get(
                f"https://raw.githubusercontent.com/{REPO}/{sha}/.github/workflows/buildusd.yml",
                timeout=60,
            )
            response.raise_for_status()
            workflow_path.write_text(response.text)
    with (ROOT / "windows_timing_artifact_samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(artifact_summary[0]))
        writer.writeheader()
        writer.writerows(artifact_summary)


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / ".cache/windows_timing_investigation"
    )
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    try:
        investigate(args.output_dir)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
