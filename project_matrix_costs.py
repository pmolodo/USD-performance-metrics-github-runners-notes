#!/usr/bin/env python

"""Project runner costs and cost shares for a configurable platform build matrix.

Print JSON to stdout. Prices are planning inputs from the parent cost report,
not live quotes. Each configuration uses one runner exclusively while active.
"""

import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path

SCENARIOS = ("low", "mid", "high")
RELEASES_PER_YEAR = 4
MONTHS_PER_YEAR = 12
MONTHS_PER_RELEASE = MONTHS_PER_YEAR / RELEASES_PER_YEAR
DAYS_PER_MONTH = 30.4375
DAYS_PER_YEAR = DAYS_PER_MONTH * MONTHS_PER_YEAR
HOURS_PER_MONTH = DAYS_PER_MONTH * 24
HOURLY_RATES = {"linux": 2.0144, "windows": 3.4864}
ARTIFACT_MIB = {"linux": 677.2, "windows": 67.3, "macos": 76.6}
MAC_PURCHASE = 1699.0
MAC_COLOCATION_MONTHLY = 59.99
MAC_RENT_MONTHLY = 349.0
RETENTION_DAYS = 90
GP3_GIB_MONTH = 0.08
EGRESS_GB_RATE = 0.09
FREE_EGRESS_GB_MONTH = 100
ANNUAL_PRICE_FACTOR = 1.03


def project_year(rows, counts, hours, utilization):
    artifact_mib = sum(counts[os] * ARTIFACT_MIB[os] for os in counts)
    artifact_gib = artifact_mib / 1024
    artifact_gb = artifact_mib * 1024**2 / 1e9
    result = {
        "counts": counts,
        "hours_per_build": hours,
        "target_mac_utilization": utilization,
        "artifact_mib_per_trigger": artifact_mib,
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        events = sum(
            float(r[f"{scenario}_events_including_reruns_per_release"]) for r in rows
        )
        monthly_peak = max(
            float(r[f"{scenario}_events_including_reruns"]) for r in rows
        )
        runner_hours = {os: events * count * hours for os, count in counts.items()}
        compute = {os: runner_hours[os] * rate for os, rate in HOURLY_RATES.items()}
        macs = math.ceil(
            monthly_peak * counts["macos"] * hours / (HOURS_PER_MONTH * utilization)
        )
        mac_purchase = macs * MAC_PURCHASE
        mac_colo = macs * MONTHS_PER_YEAR * MAC_COLOCATION_MONTHLY
        mac_rent = macs * MONTHS_PER_YEAR * MAC_RENT_MONTHLY
        retained_gib = events / DAYS_PER_YEAR * RETENTION_DAYS * artifact_gib
        storage = retained_gib * GP3_GIB_MONTH * MONTHS_PER_YEAR
        transfers = {}
        for fraction in (0.01, 0.05, 1.0):
            gb = events * artifact_gb * fraction
            paid_gb = sum(
                max(
                    0,
                    float(r[f"{scenario}_events_including_reruns_per_release"])
                    / MONTHS_PER_RELEASE
                    * artifact_gb
                    * fraction
                    - FREE_EGRESS_GB_MONTH,
                )
                * MONTHS_PER_RELEASE
                for r in rows
            )
            transfers[str(fraction)] = {
                "gb": gb,
                "gross_cost": gb * EGRESS_GB_RATE,
                "unused_allowance_cost": paid_gb * EGRESS_GB_RATE,
            }
        peak_rate = max(float(r[f"{scenario}_events_including_reruns"]) for r in rows)
        transfer = transfers["0.05"]["gross_cost"]
        common = sum(compute.values()) + storage + transfer
        result["scenarios"][scenario] = {
            "events": events,
            "runner_hours": runner_hours,
            "compute": compute,
            "mac_hosts": macs,
            "mac_purchase": mac_purchase,
            "mac_colocation": mac_colo,
            "mac_rental": mac_rent,
            "annual_rate_retained_gib": retained_gib,
            "storage": storage,
            "transfer_1_in_20": transfer,
            "hybrid": common + mac_purchase + mac_colo,
            "rented_mac": common + mac_rent,
            "transfers": transfers,
            "final_release_hours_per_month": {
                os: (
                    float(rows[-1][f"{scenario}_events_including_reruns"])
                    * count
                    * hours
                )
                for os, count in counts.items()
            },
            "peak_release_retained_gib": (
                peak_rate / DAYS_PER_MONTH * RETENTION_DAYS * artifact_gib
            ),
        }
    return result


def project(forecast, counts, hours, utilization, years):
    with forecast.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))[: years * RELEASES_PER_YEAR]
    if len(rows) != years * RELEASES_PER_YEAR:
        raise ValueError(
            f"Expected at least {years * RELEASES_PER_YEAR} release cycles"
        )
    annual = []
    owned = dict.fromkeys(SCENARIOS, 0)
    for year in range(years):
        result = project_year(
            rows[year * RELEASES_PER_YEAR : (year + 1) * RELEASES_PER_YEAR],
            counts,
            hours,
            utilization,
        )
        for scenario, row in result["scenarios"].items():
            required = row["mac_hosts"]
            added = max(0, required - owned[scenario])
            owned[scenario] += added
            row["mac_hosts"] = owned[scenario]
            row["mac_purchase"] = added * MAC_PURCHASE
            row["mac_colocation"] = (
                owned[scenario] * MONTHS_PER_YEAR * MAC_COLOCATION_MONTHLY
            )
            factor = ANNUAL_PRICE_FACTOR**year
            for key in ("mac_purchase", "mac_colocation", "mac_rental", "storage"):
                row[key] *= factor
            row["compute"] = {os: cost * factor for os, cost in row["compute"].items()}
            common = (
                sum(row["compute"].values()) + row["storage"] + row["transfer_1_in_20"]
            )
            row["hybrid"] = common + row["mac_purchase"] + row["mac_colocation"]
            row["rented_mac"] = common + row["mac_rental"]
        annual.append({"year": year + 1, "scenarios": result["scenarios"]})
    totals = {}
    for scenario in SCENARIOS:
        values = [year["scenarios"][scenario] for year in annual]
        total = {
            key: sum(row[key] for row in values)
            for key in (
                "events",
                "mac_purchase",
                "mac_colocation",
                "mac_rental",
                "storage",
                "transfer_1_in_20",
                "hybrid",
                "rented_mac",
            )
        }
        for key in ("runner_hours", "compute"):
            total[key] = {
                os: sum(row[key][os] for row in values) for os in values[0][key]
            }
        total["transfers"] = {
            fraction: {
                key: sum(row["transfers"][fraction][key] for row in values)
                for key in values[0]["transfers"][fraction]
            }
            for fraction in values[0]["transfers"]
        }
        mac_owned = total["mac_purchase"] + total["mac_colocation"]
        savings = total["rented_mac"] - total["hybrid"]
        total["savings"] = {
            "dollars": savings,
            "percent_of_rented_total": 100 * savings / total["rented_mac"],
            "percent_of_mac_rental": (
                100 * savings / total["mac_rental"] if total["mac_rental"] else None
            ),
        }
        total["cost_share_percent"] = {}
        for option, mac_cost in (
            ("hybrid", mac_owned),
            ("rented_mac", total["mac_rental"]),
        ):
            parts = {
                **total["compute"],
                "macos": mac_cost,
                "storage": total["storage"],
                "transfer": total["transfer_1_in_20"],
            }
            total["cost_share_percent"][option] = {
                part: 100 * cost / total[option] for part, cost in parts.items()
            }
        totals[scenario] = total
    return {
        "years": years,
        "counts": counts,
        "hours_per_build": hours,
        "target_mac_utilization": utilization,
        "artifact_mib_per_trigger": result["artifact_mib_per_trigger"],
        "annual": annual,
        "totals": totals,
    }


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--years",
        type=int,
        choices=range(1, 6),
        default=5,
        help="Projection horizon; each year uses four forecast releases",
    )
    parser.add_argument(
        "--forecast",
        type=Path,
        default=Path(__file__).with_name("openusd_triggering_events_forecast.csv"),
    )
    for os, count in (("linux", 6), ("windows", 5), ("macos", 5)):
        parser.add_argument(
            f"--{os}",
            type=int,
            default=count,
            help=f"Number of {os} configurations per triggering event",
        )
    parser.add_argument(
        "--hours",
        type=float,
        default=1.0,
        help="Runner hours per configuration on each platform",
    )
    parser.add_argument(
        "--mac-utilization",
        type=float,
        default=0.7,
        help="Target maximum utilization at the busiest release average",
    )
    return parser


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    counts = {os: getattr(args, os) for os in ARTIFACT_MIB}
    if min(counts.values()) < 0 or sum(counts.values()) == 0:
        parser.error(
            "Configuration counts must be nonnegative, with at least one build"
        )
    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error("--hours must be finite and positive")
    if not 0 < args.mac_utilization <= 1:
        parser.error("--mac-utilization must be in (0, 1]")
    try:
        print(
            json.dumps(
                project(
                    args.forecast, counts, args.hours, args.mac_utilization, args.years
                ),
                indent=2,
            )
        )
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
