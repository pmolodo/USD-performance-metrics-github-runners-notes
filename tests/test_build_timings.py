"""Checks for successful build timing selection and percentile semantics."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_build_timings import classify, summarize


def job(identifier="1", **updates):
    row = {
        "job_id": identifier,
        "run_id": "10",
        "name": "Linux",
        "status": "completed",
        "conclusion": "success",
        "started_at": "2026-09-01 00:00:00",
        "completed_at": "2026-09-01 01:00:00",
        "raw_json": "{}",
        "labels_json": '["ubuntu-22.04"]',
    }
    row.update(updates)
    return row


def test_inclusive_percentiles_and_single_sample():
    assert summarize([0, 10]) == {"n": 2, "p10": 1, "p50": 5, "p90": 9, "mean": 5}
    assert summarize([7]) == {"n": 1, "p10": 7, "p50": 7, "p90": 7, "mean": 7}
    assert summarize([])["p50"] == ""


def test_copied_execution_is_removed_but_real_rerun_is_retained():
    rows = classify(
        [
            job(),
            job("2"),
            job(
                "3",
                started_at="2026-09-01 02:00:00",
                completed_at="2026-09-01 03:00:00",
            ),
        ]
    )
    assert [r["filter_result"] for r in rows] == [
        "included",
        "duplicate_execution",
        "included",
    ]


def test_failed_job_invalid_timestamp_and_masked_step_failure():
    raw = json.dumps({"steps": [{"name": "Build USD", "conclusion": "failure"}]})
    rows = classify(
        [job(conclusion="failure"), job("2", completed_at=""), job("3", raw_json=raw)]
    )
    assert [r["filter_result"] for r in rows] == [
        "not_successful",
        "invalid_job_timestamps",
        "build_step_not_successful_or_missing",
    ]


def test_fast_success_is_kept_and_missing_steps_are_not_zero():
    row = classify([job(completed_at="2026-09-01 00:00:30")])[0]
    assert row["filter_result"] == "included"
    assert row["job_minutes"] == 0.5
    assert row["build_step_minutes"] == ""
