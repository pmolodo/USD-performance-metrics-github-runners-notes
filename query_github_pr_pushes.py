#!/usr/bin/env python

"""Query a github project for all times that pushes were made to a PR.
"""

import argparse
import datetime
import json
import os
import requests
import sys
import traceback
import time
from collections import deque
from typing import Optional
from tqdm import tqdm

###############################################################################
# Core functions
###############################################################################


def parse_datetime_string(dt_string: str,
                          is_github_api_format: bool = False
                          ) -> datetime.datetime:
    """
    Convert a datetime string to a timezone-aware datetime object.

    Args:
        dt_string: The datetime string to parse
        is_github_api_format: If True, converts 'Z' suffix to '+00:00'
                             If False, assumes user input and adds UTC timezone

    Returns:
        A timezone-aware datetime object
    """
    if is_github_api_format:
        # GitHub API uses 'Z' suffix which needs to be converted to '+00:00'
        dt_string = dt_string.replace('Z', '+00:00')

    dt = datetime.datetime.fromisoformat(dt_string)

    # Ensure timezone awareness - add UTC if no timezone specified
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)

    return dt


def is_timestamp_in_range(timestamp: datetime.datetime,
                          start_dt: Optional[datetime.datetime],
                          end_dt: Optional[datetime.datetime]) -> bool:
    """
    Check if a timestamp falls within the specified time range.

    Args:
        timestamp: The datetime to check
        start_dt: Optional start datetime (None means no start limit)
        end_dt: Optional end datetime (None means no end limit)

    Returns:
        True if timestamp is within range, False otherwise
    """
    return ((start_dt is None or timestamp >= start_dt) and
            (end_dt is None or timestamp <= end_dt))


class PRTask:
    """Represents a PR processing task with retry information."""

    def __init__(self, pr_item: dict, max_retries: int = 3):
        self.pr_item = pr_item
        self.pr_number = pr_item.get('number')
        self.retries_remaining = max_retries
        self.error_message: Optional[str] = None
        self.processed_data: Optional[dict] = None

    def __repr__(self):
        return (f"PRTask(PR #{self.pr_number}, "
                f"retries={self.retries_remaining})")


def process_single_pr(task: PRTask, headers: dict,
                      owner: str, project: str,
                      start_dt: Optional[datetime.datetime],
                      end_dt: Optional[datetime.datetime]) -> bool:
    """Process a single PR to extract timeline data.

    Args:
        task: The PR task to process
        headers: HTTP headers for API requests
        owner: Repository owner
        project: Repository name
        start_dt: Optional start datetime filter
        end_dt: Optional end datetime filter

    Returns:
        True if processing succeeded, False if it failed
    """
    pr_item = task.pr_item
    pr_number = pr_item.get('number')

    try:
        # Get PR creation time (always included)
        pr_created_str = pr_item.get('created_at')
        if not pr_created_str:
            task.error_message = "No creation timestamp found"
            return False

        pr_created = parse_datetime_string(pr_created_str, True)

        # Start with PR creation timestamp
        timestamps = []
        if is_timestamp_in_range(pr_created, start_dt, end_dt):
            timestamps.append(pr_created.isoformat())

        # Get timeline events
        timeline_url = (f'https://api.github.com/repos/'
                        f'{owner}/{project}/'
                        f'issues/{pr_number}/timeline')

        response = requests.get(timeline_url, headers=headers, timeout=30)
        if response.status_code != 200:
            task.error_message = (f"Timeline API returned "
                                  f"{response.status_code}: {response.text}")
            return False

        timeline_events = response.json()

        # Process timeline events for commit-related activity
        for event in timeline_events:
            if event.get('event') in ['committed', 'pushed']:
                event_time_str = event.get('created_at')
                if event_time_str:
                    event_time = parse_datetime_string(event_time_str, True)
                    if is_timestamp_in_range(event_time, start_dt, end_dt):
                        timestamps.append(event_time.isoformat())

        # Store the processed data
        task.processed_data = {
            'pr_number': pr_number,
            'title': pr_item.get('title', ''),
            'timestamps': timestamps
        }

        return True

    except requests.RequestException as e:
        task.error_message = f"Request failed: {e}"
        return False


def build_pr_search_query(owner: str, project: str,
                          start_dt: Optional[datetime.datetime],
                          end_dt: Optional[datetime.datetime],
                          is_open: bool) -> str:
    """
    Build GitHub search query to find PRs by state and date range.

    Args:
        owner: Repository owner
        project: Repository name
        start_dt: Optional start datetime for filtering
        end_dt: Optional end datetime for filtering
        is_open: True for open PRs, False for closed PRs

    Returns:
        GitHub search query string
    """
    # Base query components
    query_parts = [f'repo:{owner}/{project}', 'type:pr']

    # Format dates once
    start_date = start_dt.strftime('%Y-%m-%d') if start_dt else None
    end_date = end_dt.strftime('%Y-%m-%d') if end_dt else None

    # State-specific logic
    if is_open:
        query_parts.append('is:open')
    else:  # closed PRs
        if start_date:
            query_parts.append(f'closed:>={start_date}')
        else:
            query_parts.append('is:closed')

    # Common start date filtering (updated constraint)
    if start_date:
        query_parts.append(f'updated:>={start_date}')

    # Common end date filtering
    if end_date:
        query_parts.append(f'created:<={end_date}')

    return ' '.join(query_parts)


def build_open_prs_query(owner: str, project: str,
                         start_dt: Optional[datetime.datetime],
                         end_dt: Optional[datetime.datetime]) -> str:
    """Build GitHub search query to find open PRs."""
    return build_pr_search_query(owner, project, start_dt, end_dt, True)


def build_closed_prs_query(owner: str, project: str,
                           start_dt: Optional[datetime.datetime],
                           end_dt: Optional[datetime.datetime]) -> str:
    """Build query to find closed PRs with activity after start_dt."""
    return build_pr_search_query(owner, project, start_dt, end_dt, False)


def search_prs_with_query(query: str, headers: dict) -> list:
    """Execute a single search query and return all paginated results."""
    search_url = 'https://api.github.com/search/issues'
    all_prs = []
    page = 1
    per_page = 100

    while True:
        params = {
            'q': query,
            'page': page,
            'per_page': per_page,
            'sort': 'created',
            'order': 'desc'
        }

        response = requests.get(search_url, headers=headers, params=params,
                                timeout=30)
        if response.status_code != 200:
            print(f"Error in search API: {response.status_code} - "
                  f"{response.text}")
            break

        data = response.json()
        items = data.get('items', [])

        if not items:
            break

        all_prs.extend(items)

        # Check if we've got all results
        if len(items) < per_page:
            break

        page += 1

    return all_prs


def search_filtered_prs(owner: str, project: str, headers: dict,
                        start_dt: Optional[datetime.datetime],
                        end_dt: Optional[datetime.datetime]):
    """Use search API to get PRs filtered by date range.

    Makes two separate queries (open PRs and closed PRs) since GitHub's
    issue search API doesn't support OR operators with parentheses.
    """
    # Build separate queries for open and closed PRs
    open_query = build_open_prs_query(owner, project, start_dt, end_dt)
    closed_query = build_closed_prs_query(owner, project, start_dt, end_dt)

    print(f"Open PRs query: {open_query}")
    print(f"Closed PRs query: {closed_query}")

    # Execute both queries
    open_prs = search_prs_with_query(open_query, headers)
    closed_prs = search_prs_with_query(closed_query, headers)

    print(f"Found {len(open_prs)} open PRs and {len(closed_prs)} closed PRs")

    # Combine results and remove duplicates (shouldn't be any, but be safe)
    all_prs = open_prs + closed_prs
    seen_pr_numbers = set()
    unique_prs = []

    for pr in all_prs:
        pr_number = pr.get('number')
        if pr_number not in seen_pr_numbers:
            seen_pr_numbers.add(pr_number)
            unique_prs.append(pr)
        else:
            print(f"Warning: Duplicate PR #{pr_number} found and removed")

    print(f"Total unique PRs: {len(unique_prs)}")
    return unique_prs


def query_github_pr_pushes(owner: str, project: str,
                           start_time: Optional[str] = None,
                           end_time: Optional[str] = None,
                           token: Optional[str] = None,
                           output_file: str = "pr_push_times.json",
                           max_prs: Optional[int] = None):
    """
    Query GitHub API for PR creation and commit event timestamps.

    Tracks when PRs are created and when new commits are added to PR branches
    using the timeline API, not individual commit authoring times.

    Filters PRs by lifetime overlap with time range, and filters all collected
    timestamps to only include those within the specified time range.

    Args:
        owner: Repository owner
        project: Repository name
        start_time: Optional start time filter (ISO format string)
        end_time: Optional end time filter (ISO format string)
        token: GitHub personal access token
        output_file: Output JSON file path
        max_prs: Optional maximum number of PRs to process (for testing)
    """
    # Setup headers
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'USD-Performance-Metrics-Script'
    }

    # Add authentication if token provided
    if token:
        headers['Authorization'] = f'token {token}'
    elif 'GITHUB_TOKEN' in os.environ:
        headers['Authorization'] = f'token {os.environ["GITHUB_TOKEN"]}'
    else:
        print("Warning: No GitHub token provided. You may hit rate limits.")
        print("Consider using --token YOUR_TOKEN or setting GITHUB_TOKEN "
              "environment variable.")

    print(f"Fetching PRs for {owner}/{project}...")

    # Process time filters once outside the loop
    start_dt = None
    end_dt = None
    if start_time:
        start_dt = parse_datetime_string(start_time)
    if end_time:
        end_dt = parse_datetime_string(end_time)

    # Use search API to get filtered PRs
    print("Using search API to get PRs within date range...")
    filtered_prs = search_filtered_prs(owner, project, headers,
                                       start_dt, end_dt)

    if not filtered_prs:
        print("No PRs found matching the search criteria.")
        return

    print(f"Found {len(filtered_prs)} PRs matching criteria")

    # Apply max_prs limit if specified
    if max_prs is not None and max_prs < len(filtered_prs):
        print(f"Limiting to first {max_prs} PRs (--max-prs specified)")
        filtered_prs = filtered_prs[:max_prs]

    # Initialize deque-based retry system for PR processing
    print(f"Setting up retry system for {len(filtered_prs)} PRs...")

    # Create PR tasks and add to processing queue
    active_pr_tasks = deque()
    successful_prs = []
    failed_prs = []

    for pr_item in filtered_prs:
        task = PRTask(pr_item)
        active_pr_tasks.append(task)

    # Initialize progress tracking
    total_prs = len(filtered_prs)
    pbar = tqdm(total=total_prs, desc="Processing PRs", unit="PR")
    processed_count = 0

    # Main processing loop with retry logic
    while active_pr_tasks:
        task = active_pr_tasks.popleft()

        success = process_single_pr(task, headers, owner, project,
                                    start_dt, end_dt)

        if success:
            # Success - add to successful list and update progress
            successful_prs.append(task)
            processed_count += 1
            pbar.update(1)
            pbar.set_postfix({
                'success': len(successful_prs),
                'failed': len(failed_prs),
                'retrying': len(active_pr_tasks)
            })

        else:
            # Failed - check if we should retry
            task.retries_remaining -= 1
            if task.retries_remaining > 0:
                # Add back to queue for retry
                active_pr_tasks.append(task)
                pbar.set_description(
                    f"Processing PRs (PR #{task.pr_number} retrying)"
                )
            else:
                # Max retries exceeded - mark as failed
                failed_prs.append(task)
                processed_count += 1
                pbar.update(1)
                pbar.set_postfix({
                    'success': len(successful_prs),
                    'failed': len(failed_prs),
                    'retrying': len(active_pr_tasks)
                })
                print(f"\n  ❌ PR #{task.pr_number} failed permanently: "
                      f"{task.error_message}")

        # Small delay between requests to be nice to the API
        if active_pr_tasks or success:  # Only delay if more work to do
            time.sleep(0.5)

    # Close progress bar
    pbar.close()

    # Collect results from successful PR processing
    all_pr_data = []
    for task in successful_prs:
        if task.processed_data:
            all_pr_data.append(task.processed_data)

    print("\nPR Processing Summary:")
    print(f"  ✅ Successfully processed: {len(successful_prs)}")
    print(f"  ❌ Failed to process: {len(failed_prs)}")

    if failed_prs:
        print("\nFailed PRs:")
        for task in failed_prs:
            print(f"  - PR #{task.pr_number}: {task.error_message}")

    # Prepare processing statistics
    processing_stats = {
        'total_prs_found': len(filtered_prs),
        'successfully_processed': len(successful_prs),
        'failed_to_process': len(failed_prs),
        'failed_pr_details': [{
            'pr_number': task.pr_number,
            'error': task.error_message,
            'retries_attempted': 3 - task.retries_remaining
        } for task in failed_prs]
    }

    # Calculate total timestamp events across all PRs
    total_timestamp_events = sum(len(pr_data['timestamps'])
                                 for pr_data in all_pr_data)

    print(f"\nFound {len(all_pr_data)} PRs with push data.")
    print(f"Total timestamp events across all PRs: {total_timestamp_events}")

    # Save results to JSON file
    output_data = {
        'repository': f'{owner}/{project}',
        'query_timestamp': datetime.datetime.now().isoformat(),
        'filters': {
            'start_time': start_time,
            'end_time': end_time
        },
        'processing_statistics': processing_stats,
        'total_prs_processed': len(all_pr_data),
        'total_timestamp_events': total_timestamp_events,
        'prs': all_pr_data
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_file}")
    return output_data

###############################################################################
# CLI
###############################################################################


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("owner", help="The owner of the github project")
    parser.add_argument("project", help="The name of the github project")
    parser.add_argument("--start",
                        help="Optional start time filter "
                             "(ISO format, e.g., 2023-01-01T00:00:00)")
    parser.add_argument("--end",
                        help="Optional end time filter "
                             "(ISO format, e.g., 2023-12-31T23:59:59)")
    parser.add_argument("--token",
                        help="GitHub personal access token "
                             "(can also use GITHUB_TOKEN env var)")
    parser.add_argument("--output", default="pr_push_times.json",
                        help="Output JSON file path")
    parser.add_argument("--max-prs", type=int,
                        help="Maximum number of PRs to process (for testing)")
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = get_parser()
    args = parser.parse_args(argv)
    try:
        query_github_pr_pushes(
            owner=args.owner,
            project=args.project,
            start_time=args.start,
            end_time=args.end,
            token=args.token,
            output_file=args.output,
            max_prs=args.max_prs
        )
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
