#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

exec uv --cache-dir .cache/uv run python -u openusd_usage_dolt.py "$@"
