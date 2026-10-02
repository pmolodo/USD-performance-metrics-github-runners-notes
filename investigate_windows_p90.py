#!/usr/bin/env python

"""Inspect Windows slow-build logs and workflow changes around August 2026."""

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
API = "https://api.github.com/repos/PixarAnimationStudios/OpenUSD"


def investigate(output):
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    token = subprocess.run(
        ["gh", "auth", "token"], check=True, capture_output=True, text=True
    ).stdout.strip()
    session.headers["Authorization"] = f"Bearer {token}"
    with (ROOT / "openusd_build_timings_jobs.csv").open(newline="") as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if r["platform"] == "windows" and r["filter_result"] == "included"
        ]
    for month in ("2026-07", "2026-08", "2026-09"):
        current = sorted(
            [r for r in rows if r["month"] == month],
            key=lambda r: float(r["job_minutes"]),
            reverse=True,
        )
        print(
            "MONTH",
            month,
            "DURATIONS",
            [round(float(r["job_minutes"]), 2) for r in current],
        )
        for row in current[: 6 if month == "2026-08" else 4]:
            path = output / f"{row['job_id']}.log"
            if not path.exists():
                response = session.get(
                    API + f"/actions/jobs/{row['job_id']}/logs", timeout=60
                )
                if response.status_code in (404, 410):
                    print("LOG UNAVAILABLE", row["job_id"], response.status_code)
                    continue
                response.raise_for_status()
                path.write_text(response.text)
            log = path.read_text()
            pattern = re.compile(
                r"Cacheable calls:|\sHits:|\sMisses:|Image Version:|Image:\s|cache not"
                r" found|tests passed|Dependencies\s|Compiler\s",
                re.I,
            )
            matches = [line for line in log.splitlines() if pattern.search(line)]
            print(
                "SAMPLE",
                row["started_at"],
                row["job_minutes"],
                row["url"],
                "\n".join(matches),
            )
    for source_path in (".github/workflows/buildusd.yml", "build_scripts/build_usd.py"):
        index = output / (Path(source_path).name + "-commits.json")
        if not index.exists():
            commits = []
            url = API + "/commits"
            params = {
                "sha": "dev",
                "path": source_path,
                "since": "2026-07-01T00:00:00Z",
                "until": "2026-09-01T00:00:00Z",
                "per_page": 100,
            }
            while url:
                response = session.get(url, params=params, timeout=60)
                response.raise_for_status()
                commits.extend(response.json())
                url = response.links.get("next", {}).get("url")
                params = None
            index.write_text(json.dumps(commits, indent=2))
        for commit in json.loads(index.read_text()):
            print(
                "CHANGE",
                source_path,
                commit["sha"],
                commit["commit"]["committer"]["date"],
                commit["commit"]["message"].splitlines()[0],
            )
            detail_path = output / (commit["sha"] + ".json")
            if not detail_path.exists():
                response = session.get(API + "/commits/" + commit["sha"], timeout=60)
                response.raise_for_status()
                detail_path.write_text(json.dumps(response.json(), indent=2))
            detail = json.loads(detail_path.read_text())
            for file in detail["files"]:
                if file["filename"] == source_path:
                    print(file.get("patch", "No patch returned"))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / ".cache/windows_p90_investigation"
    )
    args = parser.parse_args(argv)
    try:
        investigate(args.output_dir)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
