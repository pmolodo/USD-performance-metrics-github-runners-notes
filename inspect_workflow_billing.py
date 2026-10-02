#!/usr/bin/env python
"""Cache public workflow billing inputs and summarize the retained Actions data."""

import argparse
import collections
import json
import subprocess
import sys
import traceback
from pathlib import Path

import requests

from analyze_build_timings import query

ROOT = Path(__file__).resolve().parent
API = "https://api.github.com/repos/PixarAnimationStudios/OpenUSD"


def inspect(output, fetch_artifacts=False):
    output.mkdir(parents=True, exist_ok=True)
    database = ROOT / ".cache/openusd_usage_dolt"
    for name, sql in {
        "jobs": (
            "SELECT j.*, r.event, r.created_at AS run_created_at FROM workflow_jobs j"
            " JOIN workflow_runs r ON r.run_id=j.run_id ORDER BY j.started_at, j.job_id"
        ),
        "runs": "SELECT * FROM workflow_runs ORDER BY created_at",
        "metadata": "SELECT * FROM sync_metadata ORDER BY meta_key",
    }.items():
        rows = query(database, sql)
        (output / f"{name}.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
        if name == "jobs":
            counts = collections.Counter(
                (r["workflow_name"], r["name"], r["labels_json"]) for r in rows
            )
            for key, count in counts.items():
                print(count, key)
        elif name == "runs":
            print("RUNS", len(rows), rows[0]["created_at"], rows[-1]["created_at"])
            print("MISSING JOB FETCH", sum(not r["jobs_fetched_at"] for r in rows))
        else:
            print("METADATA", rows)
    if fetch_artifacts:
        session = requests.Session()
        token = subprocess.run(
            ["gh", "auth", "token"], check=True, capture_output=True, text=True
        ).stdout.strip()
        session.headers["Authorization"] = f"Bearer {token}"
        artifacts = []
        url = API + "/actions/artifacts?per_page=100"
        while url:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            artifacts.extend(response.json()["artifacts"])
            url = response.links.get("next", {}).get("url")
        (output / "artifacts.json").write_text(
            json.dumps(artifacts, indent=2, sort_keys=True)
        )
        print(
            "ARTIFACTS",
            len(artifacts),
            collections.Counter(a["name"] for a in artifacts),
        )


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / ".cache/workflow_billing"
    )
    parser.add_argument("--fetch-artifacts", action="store_true")
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    try:
        inspect(args.output_dir, args.fetch_artifacts)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
