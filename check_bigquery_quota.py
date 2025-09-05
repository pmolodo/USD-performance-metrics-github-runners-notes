#!/usr/bin/env python3
"""
Quick script to check BigQuery quota usage.
"""

import argparse
import inspect
import os
import sys

from google.cloud import bigquery

###############################################################################
# Constants
###############################################################################

THIS_FILE = os.path.abspath(inspect.getsourcefile(lambda: None) or __file__)
THIS_DIR = os.path.dirname(THIS_FILE)

# Import bigquery_utils with fallback to add THIS_DIR to sys.path
try:
    import bigquery_utils
except ImportError:
    # Add THIS_DIR to sys.path and try again
    sys.path.insert(0, THIS_DIR)
    import bigquery_utils


def check_quota():
    """Check current BigQuery quota usage and limits."""
    try:
        client = bigquery.Client()
        project_id = client.project

        print(f"=== BigQuery Quota for Project: {project_id} ===")

        # Get recent jobs to estimate current month usage
        jobs = list(client.list_jobs(max_results=100))

        monthly_bytes = 0
        query_count = 0

        # Calculate usage from recent queries
        for job in jobs:
            if job.job_type == "query" and job.state == "DONE":
                job_result = client.get_job(job.job_id)
                bytes_processed = getattr(job_result, "total_bytes_processed", 0) or 0
                monthly_bytes += bytes_processed
                query_count += 1

        monthly_gb = monthly_bytes / (1024**3)
        monthly_tb = monthly_gb / 1024

        print(f"Recent usage (from {query_count} queries):")
        print(f"  Total bytes: {monthly_bytes:,}")
        print(f"  Total GB: {monthly_gb:.2f}")
        print(f"  Total TB: {monthly_tb:.4f}")
        print()

        # Free tier limits
        free_tb_limit = 1.0  # 1 TB per month free
        remaining_tb = free_tb_limit - monthly_tb

        print("Free Tier Limits:")
        print(f"  Monthly free quota: {free_tb_limit} TB")
        print(f"  Estimated remaining: {remaining_tb:.4f} TB")

        if remaining_tb <= 0:
            print("  🚨 QUOTA EXCEEDED! No free quota remaining.")
            print("  💡 Options:")
            print("     - Wait until next month (quota resets on 1st)")
            print("     - Create new Google Cloud project")
            print("     - Enable billing (first 1TB still free)")
            print()
            print("  📅 Quota resets on the 1st of each month")
        elif remaining_tb < 0.1:
            print(f"  ⚠️  WARNING: Only {remaining_tb:.4f} TB remaining!")
        else:
            print(f"  ✅ You have {remaining_tb:.4f} TB remaining")

        print()
        print("💡 Pro tip: Use dry runs to estimate query costs:")
        print("   job_config = bigquery.QueryJobConfig(dry_run=True)")

        return {
            "used_tb": monthly_tb,
            "remaining_tb": remaining_tb,
            "query_count": query_count,
            "project_id": project_id,
        }

    except Exception as e:  # pylint: disable=broad-except
        print(f"Error checking quota: {e}")
        print("Manual check: https://console.cloud.google.com/bigquery")
        return None


def get_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add BigQuery-related arguments
    bigquery_utils.add_bigquery_args(parser)

    return parser


def main(argv=None):
    """Main function with argument parsing and credential setup."""
    if argv is None:
        argv = sys.argv[1:]

    parser = get_parser()
    args = parser.parse_args(argv)

    try:
        # Set up Google Cloud credentials
        bigquery_utils.setup_credentials(args.credentials_file)

        # Check quota
        result = check_quota()
        return 0 if result else 1

    except Exception as e:  # pylint: disable=broad-except
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
