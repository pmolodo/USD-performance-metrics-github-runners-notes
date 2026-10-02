#!/usr/bin/env python

"""Inspect recent Linux/macOS cache restoration and actual build commands."""

import argparse
import csv
import re
import subprocess
import sys
import traceback
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
REPOSITORY = "PixarAnimationStudios/OpenUSD"
REFERENCE_SHA = "240e3ddef8d79a5800490c85ca4a7741397f8516"


def inspect(output):
    with (ROOT / "openusd_build_timings_jobs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    session = requests.Session()
    token = subprocess.run(
        ["gh", "auth", "token"], check=True, capture_output=True, text=True
    ).stdout.strip()
    session.headers["Authorization"] = f"Bearer {token}"
    output.mkdir(parents=True, exist_ok=True)
    for platform in ("linux", "macos"):
        row = [
            r
            for r in rows
            if r["platform"] == platform and r["filter_result"] == "included"
        ][-1]
        print(
            platform,
            row["url"],
            "job minutes",
            row["job_minutes"],
            "build minutes",
            row["build_step_minutes"],
        )
        path = output / f"{row['job_id']}.log"
        if not path.exists():
            response = session.get(
                f"https://api.github.com/repos/{REPOSITORY}/actions/jobs/{row['job_id']}/logs",
                timeout=60,
            )
            response.raise_for_status()
            path.write_text(response.text)
        log = path.read_text()
        pattern = re.compile(
            r"cache restored|cache hit|cache not found|dependencies to"
            r" build|STATUS:|ccache|CMAKE_.*COMPILER_LAUNCHER|Build Succeeded|tests"
            r" passed",
            re.I,
        )
        matches = [line for line in log.splitlines() if pattern.search(line)]
        print("\n".join(matches[:35] + matches[-8:]))
        compilation = [
            line
            for line in log.splitlines()
            if "Building CXX object" in line or "CompileC " in line
        ]
        print("Compilation lines:", len(compilation), "Examples:", compilation[:2])
    build_path = output / f"{REFERENCE_SHA}-build_usd.py"
    if not build_path.exists():
        response = requests.get(
            f"https://raw.githubusercontent.com/{REPOSITORY}/{REFERENCE_SHA}/build_scripts/build_usd.py",
            timeout=60,
        )
        response.raise_for_status()
        build_path.write_text(response.text)
    lines = build_path.read_text().splitlines()
    for number, line in enumerate(lines):
        if "dependenciesToBuild + [USD]" in line:
            print(
                "Build script:",
                number + 1,
                "\n".join(lines[max(0, number - 2) : number + 8]),
            )


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
        inspect(args.output_dir)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
