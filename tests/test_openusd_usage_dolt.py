#!/usr/bin/env python

"""Tests for release-cycle OpenUSD usage summaries."""

import datetime
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openusd_usage_dolt


def release(tag_name: str, published_at: str) -> dict[str, str]:
    return {
        "html_url": f"https://example.com/{tag_name}",
        "name": tag_name,
        "published_at": published_at,
        "tag_name": tag_name,
    }


def test_cadence_release_pattern_excludes_patch_and_preview_tags():
    pattern = openusd_usage_dolt.CADENCE_RELEASE_PATTERN

    assert pattern.fullmatch("v26.08")
    assert not pattern.fullmatch("v25.05.01")
    assert not pattern.fullmatch("v25.02a")


def test_release_cycles_flag_long_release_and_current_period():
    releases = [
        release("v25.05", "2025-05-01 00:00:00"),
        release("v25.08", "2025-08-01 00:00:00"),
        release("v25.11", "2025-11-01 00:00:00"),
        release("v26.03", "2026-03-15 00:00:00"),
    ]

    cycles = openusd_usage_dolt.release_cycles(
        releases,
        first_observed=datetime.datetime(2025, 7, 1),
        data_end=datetime.datetime(2026, 4, 1),
    )

    assert [cycle["release_cycle"] for cycle in cycles] == [
        "v25.08",
        "v25.11",
        "v26.03",
        "post-v26.03",
    ]
    assert cycles[0]["coverage_status"] == "partial"
    assert cycles[1]["coverage_status"] == "partial"
    assert cycles[2]["coverage_status"] == "complete"
    assert cycles[2]["significantly_long_cycle"] is True
    assert cycles[3]["coverage_status"] == "in_progress"


def test_core_build_family_accepts_only_exact_summary_jobs():
    base_job = {"workflow_name": "BuildUSD"}

    assert (
        openusd_usage_dolt.core_build_family({**base_job, "name": "Linux"}) == "linux"
    )
    assert (
        openusd_usage_dolt.core_build_family({**base_job, "name": "macOS"}) == "macos"
    )
    assert (
        openusd_usage_dolt.core_build_family({**base_job, "name": "Windows"})
        == "windows"
    )
    assert (
        openusd_usage_dolt.core_build_family({**base_job, "name": "Wasm (Wasm)"})
        is None
    )
    assert (
        openusd_usage_dolt.core_build_family({**base_job, "name": "GPUTests"}) is None
    )
    assert (
        openusd_usage_dolt.core_build_family(
            {"workflow_name": "PyPiPackaging", "name": "Linux"}
        )
        is None
    )


def test_seasonal_forecast_preserves_cycle_total():
    values = openusd_usage_dolt.seasonal_forecast_values(67, 0.5, months=3)

    assert [value["release_phase"] for value in values] == [
        "early",
        "middle",
        "late",
    ]
    assert sum(value["seasonal_value"] for value in values) == pytest.approx(
        sum(value["trend_level"] for value in values)
    )
    assert values[0]["seasonal_value"] < values[1]["seasonal_value"]
    assert values[1]["seasonal_value"] < values[2]["seasonal_value"]


def test_release_forecast_continues_observed_trend():
    actual_rows = [
        {
            "coverage_status": "partial",
            "events_including_reruns_per_30_4375_days": 999,
        },
        {
            "coverage_status": "complete",
            "events_including_reruns_per_30_4375_days": 10,
        },
        {
            "coverage_status": "complete",
            "events_including_reruns_per_30_4375_days": 20,
        },
        {
            "coverage_status": "in_progress",
            "events_including_reruns_per_30_4375_days": 30,
        },
    ]

    rows = openusd_usage_dolt.forecast_release_rows(actual_rows, 2)

    assert rows[0]["low_events_including_reruns"] == pytest.approx(32.5)
    assert rows[0]["mid_events_including_reruns"] == pytest.approx(35)
    assert rows[0]["high_events_including_reruns"] == pytest.approx(40)
    assert rows[1]["high_events_including_reruns"] == pytest.approx(50)
