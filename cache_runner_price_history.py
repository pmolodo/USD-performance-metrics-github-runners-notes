#!/usr/bin/env python
"""Cache GitHub's published runner prices from before the January 2026 change."""

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent


def cache(output):
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    token = subprocess.run(
        ["gh", "auth", "token"], check=True, capture_output=True, text=True
    ).stdout.strip()
    session.headers["Authorization"] = f"Bearer {token}"
    path = "content/billing/reference/actions-runner-pricing.md"
    response = session.get(
        "https://api.github.com/repos/github/docs/commits",
        params={"path": path, "until": "2025-12-01T00:00:00Z", "per_page": 1},
        timeout=60,
    )
    response.raise_for_status()
    commit = response.json()[0]
    (output / "price_commit.json").write_text(
        json.dumps(commit, indent=2, sort_keys=True)
    )
    sha = commit["sha"]
    response = requests.get(
        f"https://raw.githubusercontent.com/github/docs/{sha}/{path}", timeout=60
    )
    response.raise_for_status()
    (output / "runner_prices_2025.md").write_text(response.text)
    print(sha, response.text)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / ".cache/workflow_billing"
    )
    args = parser.parse_args(argv)
    try:
        cache(args.output_dir)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
