#!/usr/bin/env python

"""Locate the commit introducing persistent Windows compiler caching."""

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
API = "https://api.github.com/repos/PixarAnimationStudios/OpenUSD"


def find_change(output):
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    token = subprocess.run(
        ["gh", "auth", "token"], check=True, capture_output=True, text=True
    ).stdout.strip()
    session.headers["Authorization"] = f"Bearer {token}"
    index = output / "workflow_commits.json"
    if not index.exists():
        commits = []
        url = API + "/commits"
        params = {
            "sha": "dev",
            "path": ".github/workflows/buildusd.yml",
            "since": "2026-01-01T00:00:00Z",
            "until": "2026-07-01T00:00:00Z",
            "per_page": 100,
        }
        while url:
            response = session.get(url, params=params, timeout=60)
            response.raise_for_status()
            commits.extend(response.json())
            url = response.links.get("next", {}).get("url")
            params = None
        index.write_text(json.dumps(commits, indent=2))
    commits = json.loads(index.read_text())
    for commit in commits:
        print(
            commit["sha"],
            commit["commit"]["committer"]["date"],
            commit["commit"]["message"].splitlines()[0],
        )
        path = output / f"{commit['sha']}.json"
        if not path.exists():
            response = session.get(API + "/commits/" + commit["sha"], timeout=60)
            response.raise_for_status()
            path.write_text(json.dumps(response.json(), indent=2))
        detail = json.loads(path.read_text())
        for file in detail["files"]:
            patch = file.get("patch", "")
            if (
                file["filename"] == ".github/workflows/buildusd.yml"
                and "ccache" in patch
            ):
                print(
                    "MATCH",
                    detail["html_url"],
                    "author",
                    detail["commit"]["author"]["date"],
                    "committer",
                    detail["commit"]["committer"]["date"],
                )
                print(patch)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / ".cache/windows_cache_history"
    )
    args = parser.parse_args(argv)
    try:
        find_change(args.output_dir)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
