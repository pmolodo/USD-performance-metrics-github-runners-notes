"""Billing selection, rate changes, and retention-window checks."""

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_workflow_costs import (
    executions,
    forecast_storage,
    rate,
    retained_gib_days,
    storage_cost,
)


def job(identifier="1", **changes):
    result = {
        "job_id": identifier,
        "run_id": "1",
        "workflow_name": "BuildUSD",
        "name": "Linux",
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "started_at": "2026-08-01 00:00:00",
        "completed_at": "2026-08-01 00:01:01",
        "runner_name": "GitHub Actions 1",
        "labels_json": '["ubuntu-22.04"]',
        "raw_json": "{}",
    }
    result.update(changes)
    return result


def test_each_failed_or_cancelled_job_rounds_up():
    for conclusion in ("success", "failure", "cancelled"):
        row = executions([job(conclusion=conclusion)])[0]
        assert row["billed_minutes"] == 2
        assert row["private_equivalent_compute_usd"] == pytest.approx(0.024)
        assert row["public_compute_usd"] == 0


def test_copied_jobs_removed_and_real_rerun_kept():
    rows = executions(
        [
            job(),
            job("2", raw_json=json.dumps({"run_attempt": 2})),
            job(
                "3",
                started_at="2026-08-01 01:00:00",
                completed_at="2026-08-01 01:01:01",
                raw_json=json.dumps({"run_attempt": 2}),
            ),
        ]
    )
    assert [r["filter_result"] for r in rows] == [
        "included",
        "copied_execution",
        "included",
    ]
    assert sum(r["billed_minutes"] for r in rows) == 4


def test_skipped_unassigned_and_incomplete_are_not_charged():
    rows = executions(
        [
            job(conclusion="skipped"),
            job("2", runner_name=""),
            job("3", status="in_progress"),
        ]
    )
    assert sum(r["billed_minutes"] for r in rows) == 0


def test_historical_rates_and_paid_public_gpu():
    assert rate("linux", "2025-12-31") == 0.016
    assert rate("linux", "2026-01-01") == 0.012
    assert rate("windows", "2026-01-01", standard=True) == 0.010
    row = executions(
        [job(labels_json='["enterprise-linux-x64-t4gpu-4core-16vram-28ram-176ssd"]')]
    )[0]
    assert row["public_compute_usd"] == pytest.approx(0.104)


def test_storage_uses_actual_calendar_months_and_clips_horizon():
    upload = {
        "gib": 2,
        "created_at": dt.datetime(2026, 1, 1),
        "expires_at": dt.datetime(2026, 4, 1),
    }
    assert storage_cost(
        [upload], dt.datetime(2026, 1, 1), dt.datetime(2026, 3, 1)
    ) == pytest.approx(1.0)
    assert storage_cost([upload], dt.datetime(2026, 5, 1), dt.datetime(2026, 6, 1)) == 0


def test_retention_integral_includes_opening_stock_and_partial_final_uploads():
    assert retained_gib_days(-90, 0, 365.25) == pytest.approx(4050)
    assert retained_gib_days(0, 90, 90) == pytest.approx(4050)
    assert retained_gib_days(0, 90, 180) == pytest.approx(8100)


def test_constant_rate_forecast_stays_in_steady_state():
    releases = [{"mid_events_including_reruns_per_release": 91.3125}] * 4
    # One event/day, one GiB/event: 90 GiB retained for twelve months.
    assert forecast_storage(releases, "mid", 1, 1, 1) == pytest.approx(270)
