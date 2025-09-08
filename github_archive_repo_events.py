#!/usr/bin/env python

"""Download historical event data from the GitHub Archive for given repositories.

Downloads data in per-month chunks, which are saved to a local directory.
Multiple repositories can be processed in a single BigQuery call for efficiency.

Usage examples:
    # Download current month's data for OpenUSD
    python3 github_archive_repo_events.py PixarAnimationStudios/OpenUSD

    # Download for multiple repositories
    python3 github_archive_repo_events.py PixarAnimationStudios/OpenUSD microsoft/TypeScript

    # Download July 2025 data
    python3 github_archive_repo_events.py PixarAnimationStudios/OpenUSD --start-month 2025-07 --end-month 2025-07

    # Download data from July 2025 to September 2025
    python3 github_archive_repo_events.py PixarAnimationStudios/OpenUSD --start-month 2025-07 --end-month 2025-09

    # Download to custom directory
    python3 github_archive_repo_events.py PixarAnimationStudios/OpenUSD --output-dir my_data

Note: This script requires Google Cloud credentials and BigQuery access.
Set up Application Default Credentials with: gcloud auth application-default login
"""

import argparse
import dataclasses
import datetime
import inspect
import json
import os
import sys
import traceback

from collections.abc import Iterable

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

# GitHub Archive earliest available data
EARLIEST_ARCHIVE_DATE = datetime.date(2011, 2, 11)

###############################################################################
# Data Classes
###############################################################################


@dataclasses.dataclass(frozen=True, order=True)
class Month:
    year: int
    month: int

    def __post_init__(self):
        if not (1 <= self.month <= 12):
            raise ValueError(f"Month must be in range 1-12, got {self.month}")

    @classmethod
    def from_datetime(cls, dt: datetime.datetime | datetime.date) -> "Month":
        return cls(year=dt.year, month=dt.month)

    def next_month(self) -> "Month":
        """Get the next month."""
        if self.month == 12:
            return Month(self.year + 1, 1)
        else:
            return Month(self.year, self.month + 1)

    def __str__(self) -> str:
        return f"{self.year}-{self.month:02d}"


EARLIEST_ARCHIVE_MONTH = Month.from_datetime(EARLIEST_ARCHIVE_DATE)
CURRENT_MONTH = Month.from_datetime(datetime.date.today())


###############################################################################
# Global Variables
###############################################################################

# Global BigQuery client cache
_bigquery_client = None

###############################################################################
# Core functions
###############################################################################


def get_bigquery_client(credentials_file_pattern=None):
    """
    Get a BigQuery client, initializing credentials only on first use.

    Args:
        credentials_file_pattern: Optional credentials file pattern to use

    Returns:
        bigquery.Client: Initialized BigQuery client
    """
    global _bigquery_client

    if _bigquery_client is None:
        bigquery_utils.setup_credentials(credentials_file_pattern)
        _bigquery_client = bigquery.Client()

    return _bigquery_client


def decode_json_fields(event_dict):
    """
    Decode JSON string fields (payload and other) in an event dictionary.

    Args:
        event_dict: Dictionary representing a GitHub Archive event

    Returns:
        bool: True if any changes were made, False otherwise

    Note:
        Modifies the input dictionary in place.
        Prints warnings for any JSON decoding errors but preserves original values.
    """
    changes_made = False
    for field_name in ["payload", "other"]:
        if field_name in event_dict and isinstance(event_dict[field_name], str):
            try:
                event_dict[field_name] = json.loads(event_dict[field_name])
                changes_made = True
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Failed to parse {field_name} as JSON for event: {e}")
                # Keep the original string value
    return changes_made


def recursive_dict_merge(target_dict, source_dict):
    """
    Recursively merge source_dict into target_dict.

    Args:
        target_dict: Dictionary to merge into (modified in place)
        source_dict: Dictionary to merge from

    Raises:
        ValueError: If a key exists in both dictionaries but is not a dict in both
                   AND the values are not equal

    Note:
        Modifies target_dict in place. For nested dictionaries, the merge
        is recursive. Allows equal values even when not both dicts, raises
        an error only for actual conflicts (unequal non-dict values).
    """
    for key, value in source_dict.items():
        if key in target_dict:
            # Key exists in both dicts
            if isinstance(target_dict[key], dict) and isinstance(value, dict):
                # Both values are dictionaries, merge recursively
                recursive_dict_merge(target_dict[key], value)
            elif target_dict[key] == value:
                # Values are equal, no conflict - keep existing value
                pass
            else:
                # At least one value is not a dict and they're not equal, this is an error
                raise ValueError(
                    f"Key '{key}' exists in both dictionaries but is not a dict in"
                    f" both. Target type: {type(target_dict[key])}, Source type:"
                    f" {type(value)}"
                )
        else:
            # Key doesn't exist in target, add it
            target_dict[key] = value


def read_repo_events(json_filename):
    """
    Read repository events from a JSON file and transform them to be similar to GitHub REST API output.

    This function merges the "other" dictionary into the main event dictionary, while keeping
    the "payload" field as-is (since GitHub REST API events also have a "payload" field).

    Args:
        json_filename: Path to the JSON file containing repository events data

    Returns:
        dict: The modified data structure with only "other" fields merged into events

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        KeyError: If the expected structure (events key) is not found
    """
    # Read the JSON file
    with open(json_filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure we have events to process
    if "events" not in data:
        raise KeyError(f"No 'events' key found in {json_filename}")

    # Process each event to merge only "other" into the main event dict
    # Keep "payload" as-is since GitHub REST API events also have "payload"
    for event in data["events"]:
        # Merge other dictionary into main event dict
        if "other" in event and isinstance(event["other"], dict):
            other = event.pop("other")
            recursive_dict_merge(event, other)

    return data


def check_query_bytes_processed(query_sql, credentials_file_pattern=None):
    """
    Estimate query bytes processed without running it (dry run).

    Args:
        query_sql: The SQL query to estimate
        credentials_file_pattern: Optional credentials file pattern to use

    Returns:
        Number of bytes that would be processed
    """
    client = get_bigquery_client(credentials_file_pattern)
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

    # This will estimate bytes without running the query
    job = client.query(query_sql, job_config=job_config)

    return job.total_bytes_processed


def get_github_archive_month_table_name(month: Month) -> str:
    """
    Get the name of the GitHub Archive month table for the given month.
    """
    return f"githubarchive.month.{month.year}{month.month:02d}"


def get_repo_events_month_query(repo_owner: str, repo_name: str, month: Month) -> str:
    """
    Generate SQL query for all repo events for the given repo and month.

    Args:
        repo_owner: Repository owner (e.g., "PixarAnimationStudios")
        repo_name: Repository name (e.g., "OpenUSD")
        month: Month to query

    Returns:
        str: SQL query string for BigQuery
    """
    if month < EARLIEST_ARCHIVE_MONTH:
        raise ValueError(
            f"Github Archive data only available starting on {EARLIEST_ARCHIVE_MONTH}"
        )
    elif month > CURRENT_MONTH:
        raise ValueError(
            f"Github Archive data only available up to current month (got: {month})"
        )

    table = get_github_archive_month_table_name(month)

    query = f"""
    SELECT *
    FROM `{table}`
    WHERE
        repo.name = '{repo_owner}/{repo_name}'
    """

    return query


def get_multi_repo_events_month_query(repos: Iterable[str], month: Month) -> str:
    """
    Generate SQL query for all repo events for multiple repos and a given month.

    Args:
        repos: Iterable of repository full names in 'owner/repo' format
        month: Month to query

    Returns:
        str: SQL query string for BigQuery
    """
    if month < EARLIEST_ARCHIVE_MONTH:
        raise ValueError(
            f"Github Archive data only available starting on {EARLIEST_ARCHIVE_MONTH}"
        )
    elif month > CURRENT_MONTH:
        raise ValueError(
            f"Github Archive data only available up to current month (got: {month})"
        )

    table = get_github_archive_month_table_name(month)

    # Build WHERE clause for multiple repos
    repo_conditions = []
    for repo in repos:
        repo_conditions.append(f"repo.name = '{repo}'")

    where_clause = " OR ".join(repo_conditions)

    query = f"""
    SELECT *
    FROM `{table}`
    WHERE
        ({where_clause})
    """

    return query


def download_repo_events(
    repos: list[str],
    start_month: Month | None = None,
    end_month: Month | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    fix_existing_files: bool = False,
    credentials_file_pattern: str | None = None,
):
    """
    Download repository events from GitHub Archive for the given date range.

    Args:
        repos: List of repository full names in 'owner/repo' format (e.g., ["PixarAnimationStudios/OpenUSD"])
        start_month: Start month (defaults to current month if None)
        end_month: End month (defaults to current month if None)
        output_dir: Directory to save downloaded files
        fix_existing_files: If True, also fix JSON parsing in existing files
        credentials_file_pattern: Optional credentials file pattern to use

    Returns:
        list: List of on-disk file paths for all the months in the range (whether
            they were downloaded or already existed); note that the size may be less
            than the number of months if the user cancelled or there were errors.
    """

    # Validate repository format
    for repo in repos:
        if "/" not in repo:
            raise ValueError(f"Repository '{repo}' is not in 'owner/repo' format")
        if repo.count("/") != 1:
            raise ValueError(
                f"Repository '{repo}' should contain exactly one '/'. Got:"
                f" {repo.count('/')}"
            )
        owner, name = repo.split("/", 1)
        if not owner or not name:
            raise ValueError(f"Repository '{repo}' has empty owner or name component")

    # Set default months if not provided
    if start_month is None:
        start_month = CURRENT_MONTH
    if end_month is None:
        end_month = CURRENT_MONTH

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate list of months to process
    months_to_process = []
    current_month = start_month

    while current_month <= end_month:
        months_to_process.append(current_month)
        current_month = current_month.next_month()

    if not months_to_process:
        raise ValueError("No months to process")

    # Reverse the list to process months from newest to oldest
    months_to_process.reverse()

    if fix_existing_files:
        print(
            f"Scanning and fixing existing files for {len(repos)} repositories in"
            f" {output_dir}..."
        )
        total_fixed = fix_existing_json_files(output_dir)
        print(f"Fixed {total_fixed} files")
        return []

    if len(months_to_process) == 1:
        print(f"Planning to download 1 month of data for {len(repos)} repositories")
    else:
        print(
            f"Planning to download {len(months_to_process)} months of data for"
            f" {len(repos)} repositories"
        )

    print(f"Repositories: {', '.join(repos)}")
    print(f"Month range: {start_month} to {end_month}")
    print(f"Output directory: {output_dir}")
    print()

    # Check for existing files and build queries for missing ones
    # Structure: queries[(month)] = (repos_needed_for_month, query, bytes_processed, filepaths_by_repo)
    queries = {}
    existing_files = []
    total_bytes = 0

    print(
        "Checking for existing files and estimating bytes scanned for missing ones..."
    )

    for month in months_to_process:
        filepaths_by_repo = {}  # Only store repos that need downloading

        # Check each repo for this month
        for repo in repos:
            owner, name = repo.split("/", 1)
            filename = f"{owner}_{name}_{month}.json"
            filepath = os.path.join(output_dir, filename)

            if os.path.exists(filepath):
                existing_files.append(filepath)
                print(f"  {month} {repo}: File already exists, skipping")
            else:
                filepaths_by_repo[repo] = filepath  # Only store if needs downloading

        # If we need to query for any repos this month, create a single batched query
        if filepaths_by_repo:  # Check if any repos need downloading
            query = get_multi_repo_events_month_query(filepaths_by_repo, month)
            try:
                bytes_processed = check_query_bytes_processed(
                    query, credentials_file_pattern
                )
                queries[month] = (query, bytes_processed, filepaths_by_repo)
                total_bytes += bytes_processed
                print(
                    f"  {month}: {len(filepaths_by_repo)} repos,"
                    f" {bytes_processed:,} bytes ({bytes_processed/1024**3:.3f} GB)"
                )
            except Exception as e:
                print(f"  {month}: Error estimating bytes scanned - {e}")
                raise

    total_gb = total_bytes / (1024**3)
    total_tb = total_gb / 1024

    # Validate that we accounted for all repo/month combinations
    total_expected_files = len(months_to_process) * len(repos)
    total_actual_files = len(existing_files)
    for month_data in queries.values():
        _, _, filepaths_by_repo = month_data
        total_actual_files += len(filepaths_by_repo)

    if total_actual_files != total_expected_files:
        raise AssertionError(
            f"Expected {total_expected_files} total files"
            f" ({len(months_to_process)} months × {len(repos)} repos), but accounted"
            f" for {total_actual_files} files ({len(existing_files)} existing +"
            f" {total_actual_files - len(existing_files)} to download)"
        )

    # Summary of what needs to be done
    print()
    print("Summary:")
    print(f"  Total months requested: {len(months_to_process)}")
    print(f"  Files already exist: {len(existing_files)}")
    print(f"  Files to download: {len(queries)}")

    if len(queries) == 0 and not fix_existing_files:
        print("\n✅ All requested files already exist! No downloads needed.")
        return existing_files

    if len(queries) > 0:
        print()
        print("Total estimated bytes scanned for new downloads:")
        print(f"  {total_bytes:,} bytes")
        print(f"  {total_gb:.3f} GB")
        print(f"  {total_tb:.6f} TB")
        print()

        if total_tb > 0.1:  # Warn if more than 0.1 TB (100 GB)
            print(
                f"⚠️  WARNING: This will use {total_tb:.3f} TB of your BigQuery quota!"
            )
            print("Consider using a smaller date range or enabling billing.")
            print()

    # Ask for confirmation
    if fix_existing_files and len(existing_files) > 0:
        action_desc = (
            f"downloading {len(queries)} new files and fixing"
            f" {len(existing_files)} existing files"
        )
    else:
        action_desc = f"downloading {len(queries)} new files"

    response = input(f"Proceed with {action_desc}? (y/N): ")
    if response.lower() != "y":
        print("Operation cancelled.")
        print(f"Returning {len(existing_files)} existing files.")
        return existing_files

    # Process files
    downloaded_files = []
    fixed_files = []

    # Download new files
    if len(queries) > 0:
        print()
        print("Starting downloads...")
        for i, (month, (query, estimated_bytes, filepaths_by_repo)) in enumerate(
            queries.items(), 1
        ):
            print(
                f"[{i}/{len(queries)}] Downloading {month} for"
                f" {len(filepaths_by_repo)} repos..."
            )

            # Only create BigQuery client when we actually need to download data
            client = get_bigquery_client(credentials_file_pattern)

            # Execute single query for all repos this month
            query_job = client.query(query)
            results = query_job.result()

            # Group events by repository
            events_by_repo = {repo: [] for repo in filepaths_by_repo}

            for row in results:
                # Convert row to dictionary (all columns from GitHub Archive)
                event_dict = dict(row)
                # Convert datetime objects to strings for JSON serialization
                for key, value in event_dict.items():
                    if isinstance(value, datetime.datetime):
                        event_dict[key] = value.isoformat()

                # Parse JSON fields (payload and other)
                decode_json_fields(event_dict)

                # Determine which repo this event belongs to
                repo_name = event_dict.get("repo", {}).get("name")
                if repo_name in events_by_repo:
                    events_by_repo[repo_name].append(event_dict)

            # Save separate files for each repo
            for repo in filepaths_by_repo:
                owner, name = repo.split("/", 1)
                filepath = filepaths_by_repo[repo]
                events = events_by_repo[repo]

                print(f"    Saving {repo}: {len(events)} events -> {filepath}")

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "repo_owner": owner,
                            "repo_name": name,
                            "year": month.year,
                            "month": month.month,
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

    # Fix existing files if requested
    if fix_existing_files and len(existing_files) > 0:
        print()
        print("Fixing existing files...")
        for i, filepath in enumerate(existing_files, 1):
            filename = os.path.basename(filepath)
            print(f"[{i}/{len(existing_files)}] Processing {filename}...")

            try:
                # Load existing file
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "events" not in data:
                    print(f"    Warning: No 'events' field found in {filename}")
                    continue

                # Track if any changes were made to the file
                file_changed = False

                # Process each event
                for event in data["events"]:
                    if decode_json_fields(event):
                        file_changed = True

                if file_changed:
                    # Save the updated file
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    print(f"    ✅ Fixed JSON fields in {filename}")
                    fixed_files.append(filepath)
                else:
                    print(f"    ✓ No fixes needed for {filename}")

            except Exception as e:  # pylint: disable=broad-except
                print(f"    ❌ Error processing {filename}: {e}")
                continue

    all_files = existing_files + downloaded_files

    print()
    print("Process complete!")
    print(f"  Downloaded files: {len(downloaded_files)}")
    print(f"  Existing files: {len(existing_files)}")
    if fix_existing_files:
        print(f"  Files fixed: {len(fixed_files)}")
    print(f"  Total files: {len(all_files)}")
    print(f"  Output directory: {output_dir}")

    return all_files


def get_repo_events(
    repo_owner: str,
    repo_name: str,
    start_month: Month | None = None,
    end_month: Month | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    fix_existing_files: bool = False,
    credentials_file_pattern: str | None = None,
):
    """
    Get repository events by first downloading them and then reading all files.

    This function combines download_repo_events and read_repo_events to provide
    a convenient way to get all events as a flattened list.

    Args:
        repo_owner: Repository owner (e.g., "PixarAnimationStudios")
        repo_name: Repository name (e.g., "OpenUSD")
        start_month: Start month (defaults to current month if None)
        end_month: End month (defaults to current month if None)
        output_dir: Directory to save downloaded files
        fix_existing_files: If True, also fix JSON parsing in existing files
        credentials_file_pattern: Optional credentials file pattern to use

    Returns:
        list: Flattened list of all events from all months in the date range
    """
    # First, download the repo events data
    file_paths = download_repo_events(
        repo_owner=repo_owner,
        repo_name=repo_name,
        start_month=start_month,
        end_month=end_month,
        output_dir=output_dir,
        fix_existing_files=fix_existing_files,
        credentials_file_pattern=credentials_file_pattern,
    )

    # Then read events from all files and flatten into a single list
    all_events = []

    for file_path in file_paths:
        # Read events from this file
        data = read_repo_events(file_path)

        # Extract events and add to the flattened list
        if "events" in data:
            all_events.extend(data["events"])

    return all_events


###############################################################################
# CLI
###############################################################################


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "repos",
        nargs="+",
        help=(
            "Repository full names in 'owner/repo' format (e.g.,"
            " 'PixarAnimationStudios/OpenUSD'). Multiple repos can be specified."
        ),
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

    parser.add_argument(
        "--fix-existing-files",
        action="store_true",
        help=(
            "Fix JSON parsing in existing downloaded files (decode payload/other"
            " fields)"
        ),
    )

    # Add BigQuery-related arguments
    bigquery_utils.add_bigquery_args(parser)

    return parser


def parse_month_string(month_str: str) -> Month:
    """
    Parse a month string in YYYY-MM format to Month object.

    Args:
        month_str: Month string in YYYY-MM format (e.g., "2025-07")

    Returns:
        Month: Month object for the specified month

    Raises:
        ValueError: If the format is invalid
    """
    try:
        year, month = month_str.split("-")
        return Month(int(year), int(month))
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

        # Call the main function (handles both download and fix operations)
        download_repo_events(
            repos=args.repos,
            start_month=start_month,
            end_month=end_month,
            output_dir=args.output_dir,
            fix_existing_files=args.fix_existing_files,
            credentials_file_pattern=args.credentials_file,
        )

        return 0

    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
