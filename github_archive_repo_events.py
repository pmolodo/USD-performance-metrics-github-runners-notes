#!/usr/bin/env python

"""Download historical event data from the GitHub Archive for a given repository.

Downloads data in per-month chunks, which are saved to a local directory.

Usage examples:
    # Download current month's data for OpenUSD
    python3 github_archive_repo_events.py PixarAnimationStudios OpenUSD

    # Download July 2025 data
    python3 github_archive_repo_events.py PixarAnimationStudios OpenUSD --start-month 2025-07 --end-month 2025-07

    # Download data from July 2025 to September 2025
    python3 github_archive_repo_events.py PixarAnimationStudios OpenUSD --start-month 2025-07 --end-month 2025-09

    # Download to custom directory
    python3 github_archive_repo_events.py PixarAnimationStudios OpenUSD --output-dir my_data

Note: This script requires Google Cloud credentials and BigQuery access.
Set up Application Default Credentials with: gcloud auth application-default login
"""

import argparse
import datetime
import inspect
import json
import os
import sys
import traceback

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

DEFAULT_OUTPUT_DIR = os.path.join(THIS_DIR, ".cache", "github_archive_data")

###############################################################################
# Core functions
###############################################################################


def check_query_bytes_processed(query_sql):
    """
    Estimate query bytes processed without running it (dry run).

    Args:
        query_sql: The SQL query to estimate

    Returns:
        Number of bytes that would be processed
    """
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

    # This will estimate bytes without running the query
    job = client.query(query_sql, job_config=job_config)

    return job.total_bytes_processed


def get_github_archive_month_table_name(year: int, month: int) -> str:
    """
    Get the name of the GitHub Archive month table for the given year and month.
    """
    return f"githubarchive.month.{year}{month:02d}"


def get_repo_events_month_query(
    repo_owner: str, repo_name: str, year: int, month: int
) -> str:
    """
    Generate SQL query for all repo events for the given repo and month.

    Args:
        repo_owner: Repository owner (e.g., "PixarAnimationStudios")
        repo_name: Repository name (e.g., "OpenUSD")
        year: Year to query (must be >= 2011 and <= current year)
        month: Month to query (must be in range 1-12)

    Returns:
        str: SQL query string for BigQuery
    """
    now = datetime.datetime.now()
    if year < 2011:
        raise ValueError("Github Archive data only available for years >= 2011")
    elif year > now.year:
        raise ValueError(
            "Github Archive data only available for years <= current year (got:"
            f" {year})"
        )
    if month < 1 or month > 12:
        raise ValueError("Month must be in range 1-12")

    table = get_github_archive_month_table_name(year, month)

    query = f"""
    SELECT *
    FROM `{table}`
    WHERE
        repo.name = '{repo_owner}/{repo_name}'
    """

    return query


def get_repo_events(
    repo_owner: str,
    repo_name: str,
    start_month: datetime.datetime | None = None,
    end_month: datetime.datetime | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
):
    """
    Download repository events from GitHub Archive for the given date range.

    Args:
        repo_owner: Repository owner (e.g., "PixarAnimationStudios")
        repo_name: Repository name (e.g., "OpenUSD")
        start_month: Start month (defaults to current month if None)
        end_month: End month (defaults to current month if None)
        output_dir: Directory to save downloaded files

    Returns:
        list: List of file paths that were downloaded
    """
    client = bigquery.Client()

    # Set default dates if not provided
    now = datetime.datetime.now()
    if start_month is None:
        start_month = now.replace(day=1)  # First day of current month
    if end_month is None:
        end_month = now.replace(day=1)  # First day of current month

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate list of months to process
    months_to_process = []
    current = start_month.replace(day=1)  # Normalize to first day
    end = end_month.replace(day=1)

    while current <= end:
        months_to_process.append((current.year, current.month))
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    # Reverse the list to process months from newest to oldest
    months_to_process.reverse()

    print(
        f"Planning to download {len(months_to_process)} months of data for"
        f" {repo_owner}/{repo_name}"
    )
    print(
        f"Date range: {start_month.strftime('%Y-%m')} to {end_month.strftime('%Y-%m')}"
    )
    print(f"Output directory: {output_dir}")
    print()

    # Check for existing files and build queries for missing ones
    queries = []
    existing_files = []
    total_bytes = 0

    print(
        "Checking for existing files and estimating bytes scanned for missing ones..."
    )
    for year, month in months_to_process:
        # Create filename to check if it already exists
        filename = f"{repo_owner}_{repo_name}_{year}_{month:02d}.json"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"  {year}-{month:02d}: File already exists, skipping")
            existing_files.append(filepath)
        else:
            # File doesn't exist, need to query for it
            query = get_repo_events_month_query(repo_owner, repo_name, year, month)
            try:
                bytes_processed = check_query_bytes_processed(query)
                queries.append((year, month, query, bytes_processed, filepath))
                total_bytes += bytes_processed
                print(
                    f"  {year}-{month:02d}: {bytes_processed:,} bytes"
                    f" ({bytes_processed/1024**3:.3f} GB)"
                )
            except Exception as e:
                print(f"  {year}-{month:02d}: Error estimating bytes scanned - {e}")
                raise

    total_gb = total_bytes / (1024**3)
    total_tb = total_gb / 1024

    # Summary of what needs to be done
    print()
    print("Summary:")
    print(f"  Total months requested: {len(months_to_process)}")
    print(f"  Files already exist: {len(existing_files)}")
    print(f"  Files to download: {len(queries)}")

    if len(queries) == 0:
        print("\n✅ All requested files already exist! No downloads needed.")
        return existing_files

    print()
    print("Total estimated bytes scanned for new downloads:")
    print(f"  {total_bytes:,} bytes")
    print(f"  {total_gb:.3f} GB")
    print(f"  {total_tb:.6f} TB")
    print()

    if total_tb > 0.1:  # Warn if more than 0.1 TB (100 GB)
        print(f"⚠️  WARNING: This will use {total_tb:.3f} TB of your BigQuery quota!")
        print("Consider using a smaller date range or enabling billing.")
        print()

    # Ask for confirmation
    response = input(f"Proceed with downloading {len(queries)} new files? (y/N): ")
    if response.lower() != "y":
        print("Download cancelled.")
        print(f"Returning {len(existing_files)} existing files.")
        return existing_files

    # Run queries and save to disk
    downloaded_files = []

    print()
    print("Starting downloads...")
    for i, (year, month, query, estimated_bytes, filepath) in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Downloading {year}-{month:02d}...")

        # Execute query
        query_job = client.query(query)
        results = query_job.result()

        # Convert to list of dictionaries
        events = []
        for row in results:
            # Convert row to dictionary (all columns from GitHub Archive)
            event_dict = dict(row)
            # Convert datetime objects to strings for JSON serialization
            for key, value in event_dict.items():
                if isinstance(value, datetime.datetime):
                    event_dict[key] = value.isoformat()
            events.append(event_dict)

        # Save to file (filepath is already computed)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "repo_owner": repo_owner,
                    "repo_name": repo_name,
                    "year": year,
                    "month": month,
                    "download_time": datetime.datetime.now().isoformat(),
                    "event_count": len(events),
                    "estimated_bytes_processed": estimated_bytes,
                    "actual_bytes_processed": query_job.total_bytes_processed,
                    "events": events,
                },
                f,
                indent=2,
            )

        downloaded_files.append(filepath)
        print(f"  Saved {len(events)} events to {filepath}")

    # Combine existing and newly downloaded files
    all_files = existing_files + downloaded_files

    print()
    print("Process complete!")
    print(f"  Existing files: {len(existing_files)}")
    print(f"  Downloaded files: {len(downloaded_files)}")
    print(f"  Total files: {len(all_files)}")
    print(f"  Output directory: {output_dir}")

    return all_files


###############################################################################
# CLI
###############################################################################


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "repo_owner",
        help="Repository owner (e.g., 'PixarAnimationStudios')",
    )

    parser.add_argument(
        "repo_name",
        help="Repository name (e.g., 'OpenUSD')",
    )

    parser.add_argument(
        "--start-month",
        type=str,
        help=(
            "Start month in YYYY-MM format (e.g., '2025-07'). Defaults to current"
            " month."
        ),
    )

    parser.add_argument(
        "--end-month",
        type=str,
        help=(
            "End month in YYYY-MM format (e.g., '2025-07'). Defaults to current month."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save downloaded files",
    )

    # Add BigQuery-related arguments
    bigquery_utils.add_bigquery_args(parser)

    return parser


def parse_month_string(month_str: str) -> datetime.datetime:
    """
    Parse a month string in YYYY-MM format to datetime.

    Args:
        month_str: Month string in YYYY-MM format (e.g., "2025-07")

    Returns:
        datetime.datetime: First day of the specified month

    Raises:
        ValueError: If the format is invalid
    """
    try:
        year, month = month_str.split("-")
        return datetime.datetime(int(year), int(month), 1)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid month format '{month_str}'. Expected YYYY-MM (e.g., '2025-07')"
        ) from e


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = get_parser()
    args = parser.parse_args(argv)

    try:
        # Set up Google Cloud credentials
        bigquery_utils.setup_credentials(args.credentials_file)

        # Parse date arguments
        start_month = None
        end_month = None

        if args.start_month:
            start_month = parse_month_string(args.start_month)

        if args.end_month:
            end_month = parse_month_string(args.end_month)

        # Validate date range
        if start_month and end_month and start_month > end_month:
            print("Error: start_month must be <= end_month")
            return 1

        # Call the main function
        downloaded_files = get_repo_events(
            repo_owner=args.repo_owner,
            repo_name=args.repo_name,
            start_month=start_month,
            end_month=end_month,
            output_dir=args.output_dir,
        )

        if downloaded_files:
            print(f"\n✅ Successfully downloaded {len(downloaded_files)} files!")
            return 0
        else:
            print("\n❌ No files were downloaded.")
            return 1

    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
