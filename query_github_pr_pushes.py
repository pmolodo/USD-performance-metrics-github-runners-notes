#!/usr/bin/env python

"""Query a github project for all times that pushes were made to a PR.
"""

import argparse
import datetime
import json
import os
from re import S
import requests
import sys
import traceback
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Union
from tqdm import tqdm

# Constants
DEFAULT_PER_PAGE = 100
BASE_DELAY = 0.5  # seconds
MAX_EXPONENTIAL_DELAY = 300  # seconds

###############################################################################
# Data structures
###############################################################################


@dataclass
class ProcessedPr:
    """Result of successfully processing a PR."""
    pr_number: int
    title: str
    timestamps: list


@dataclass
class FailedPr:
    """Result of failed PR processing."""
    pr_number: int
    error_message: str
    rate_limit_reset_time: Optional[datetime.datetime] = None

    @property
    def is_rate_limited(self) -> bool:
        """True if this failure was due to rate limiting."""
        return self.rate_limit_reset_time is not None

    @classmethod
    def from_response(cls, pr_number: int, response) -> 'FailedPr':
        """
        Create a FailedPr from any non-200 HTTP response.

        Args:
            pr_number: PR number for the FailedPr object
            response: HTTP response object with headers, status_code, text

        Returns:
            FailedPr object with appropriate error handling
        """
        # Handle rate limiting (403/429) with special reset time parsing
        if response.status_code in [403, 429]:
            # Try x-ratelimit-reset first (primary rate limits)
            reset_ts = _get_header_int(response.headers, 'x-ratelimit-reset')
            if reset_ts is not None:
                reset_time = datetime.datetime.fromtimestamp(
                    reset_ts, tz=datetime.timezone.utc
                )
                error_message = (
                    f"Rate limited (reset at {reset_time.isoformat()}): "
                    f"{response.status_code} - {response.text}"
                )
                return cls(
                    pr_number=pr_number,
                    error_message=error_message,
                    rate_limit_reset_time=reset_time
                )

            # Try retry-after (secondary rate limits)
            retry_seconds = _get_header_int(response.headers, 'retry-after')
            if retry_seconds is not None:
                current_time = datetime.datetime.now(datetime.timezone.utc)
                delta = datetime.timedelta(seconds=retry_seconds)
                reset_time = current_time + delta
                error_message = (
                    f"Rate limited (retry after {retry_seconds}s, "
                    f"until {reset_time.isoformat()}): "
                    f"{response.status_code} - {response.text}"
                )
                return cls(
                    pr_number=pr_number,
                    error_message=error_message,
                    rate_limit_reset_time=reset_time
                )

        # Generic HTTP error (including 403/429 without rate limit headers)
        error_message = (
            f"HTTP error {response.status_code}: {response.text}"
        )
        return cls(
            pr_number=pr_number,
            error_message=error_message
        )


@dataclass
class PRTask:
    """Represents a PR processing task with retry information."""
    pr_item: dict
    retries_remaining: int = 3

    @property
    def pr_number(self) -> int:
        """Get PR number from pr_item, raising RuntimeError if missing."""
        number = self.pr_item.get('number')
        if number is None:
            raise RuntimeError("PR item is missing 'number' field")
        return number


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


def _get_header_int(headers, header_name: str) -> Optional[int]:
    """
    Safely get a header value and cast to int.

    Args:
        headers: HTTP response headers
        header_name: Name of the header to get

    Returns:
        Integer value if present and valid, None otherwise
    """
    header_value = headers.get(header_name)
    if header_value is None:
        return None

    try:
        return int(header_value)
    except (ValueError, TypeError) as e:
        print(f"Warning: Invalid {header_name} header: "
              f"{header_value!r} ({e})")
        return None


def process_single_pr(
        task: PRTask, headers: dict,
        owner: str, project: str,
        start_dt: Optional[datetime.datetime],
        end_dt: Optional[datetime.datetime]
) -> Union[ProcessedPr, FailedPr]:
    """Process a single PR to extract timeline data.

    Args:
        task: The PR task containing data to process
        headers: HTTP headers for API requests
        owner: Repository owner
        project: Repository name
        start_dt: Optional start datetime filter
        end_dt: Optional end datetime filter

    Returns:
        ProcessedPr on success, FailedPr on failure
    """
    pr_item = task.pr_item
    pr_number = task.pr_number

    try:
        # Get PR creation time (always included)
        pr_created_str = pr_item.get('created_at')
        if not pr_created_str:
            return FailedPr(
                pr_number=pr_number,
                error_message="No creation timestamp found"
            )

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

        # Handle any non-200 response (including rate limiting)
        if response.status_code != 200:
            return FailedPr.from_response(pr_number, response)

        timeline_events = response.json()

        # Process timeline events for commit-related activity
        for event in timeline_events:
            if event.get('event') in ['committed', 'pushed']:
                event_time_str = event.get('created_at')
                if event_time_str:
                    event_time = parse_datetime_string(event_time_str, True)
                    if is_timestamp_in_range(event_time, start_dt, end_dt):
                        timestamps.append(event_time.isoformat())

        # Return successful result
        return ProcessedPr(
            pr_number=pr_number,
            title=pr_item.get('title', ''),
            timestamps=timestamps
        )

    except requests.RequestException as e:
        return FailedPr(
            pr_number=pr_number,
            error_message=f"Request failed: {e}"
        )


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


def search_prs_with_query(query: str, headers: dict,
                          query_type: str = "PRs",
                          max_results: Optional[int] = None) -> list:
    """Execute a single search query and return all paginated results."""
    # Validate max_results is an int >= 0. If 0, exit immediately.
    if max_results is not None:
        if not isinstance(max_results, int) or max_results < 0:
            raise ValueError("max_results must be an integer >= 0")
        if max_results == 0:
            return []

    search_url = 'https://api.github.com/search/issues'
    all_prs = []
    page = 1

    # Initialize progress bar for search pagination
    pbar = tqdm(desc=f"Searching {query_type}", unit="PR")
    total_count = None

    while True:
        # Calculate per_page and remaining_results only if max_results is set
        if max_results is None:
            per_page = DEFAULT_PER_PAGE
        else:
            remaining_results = max_results - len(all_prs)
            per_page = min(remaining_results, DEFAULT_PER_PAGE)

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

        # Get total count from first response to configure progress bar
        if total_count is None:
            total_count = data.get('total_count', 0)
            if max_results is not None:
                total_count = min(total_count, max_results)
            pbar.total = total_count
            pbar.set_description(f"Searching {query_type} (0/{total_count})")

        if not items:
            break

        all_prs.extend(items)

        # Update progress bar to show actual PR progress
        pbar.n = len(all_prs)
        desc = f"Searching {query_type} ({len(all_prs)}/{total_count})"
        pbar.set_description(desc)
        pbar.refresh()

        if max_results is not None and len(all_prs) >= max_results:
            all_prs = all_prs[:max_results]
            break

        # Check if we've got all results
        if len(items) < per_page:
            break

        page += 1

    # Final update to show completion and leave visible
    pbar.n = len(all_prs)
    pbar.set_description(f"Found {len(all_prs)} {query_type}")
    pbar.refresh()
    # Don't close the progress bar - leave it visible
    return all_prs


def search_filtered_prs(owner: str, project: str, headers: dict,
                        start_dt: Optional[datetime.datetime],
                        end_dt: Optional[datetime.datetime],
                        max_prs: Optional[int] = None):
    """Use search API to get PRs filtered by date range.

    Makes two separate queries (open PRs and closed PRs) since GitHub's
    issue search API doesn't support OR operators with parentheses.
    """
    # Build separate queries for open and closed PRs
    open_query = build_open_prs_query(owner, project, start_dt, end_dt)
    closed_query = build_closed_prs_query(owner, project, start_dt, end_dt)

    print(f"Open PRs query: {open_query}")
    print(f"Closed PRs query: {closed_query}")

    # If max_prs is specified, we need to split between queries intelligently
    # We'll start with open PRs, then get closed PRs up to remaining limit
    remaining_limit = max_prs

    # Execute open PRs query first
    open_prs = search_prs_with_query(open_query, headers, "open PRs",
                                     max_results=remaining_limit)

    # Update remaining limit for closed PRs
    if remaining_limit is not None:
        remaining_limit -= len(open_prs)

    # Execute closed PRs query with remaining limit
    closed_prs = []
    if remaining_limit is None or remaining_limit > 0:
        closed_prs = search_prs_with_query(closed_query, headers, "closed PRs",
                                           max_results=remaining_limit)

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
                                       start_dt, end_dt, max_prs)

    if not filtered_prs:
        print("No PRs found matching the search criteria.")
        return

    print(f"Found {len(filtered_prs)} PRs matching criteria")

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

    # Main processing loop with intelligent rate limiting
    # Track consecutive non-rate-limit errors globally for exponential backoff
    consecutive_errors = 0

    while active_pr_tasks:
        task = active_pr_tasks.popleft()

        # Process the PR and get result
        result = process_single_pr(
            task, headers, owner, project, start_dt, end_dt
        )

        # Initialize sleep time for this iteration
        sleep_time = BASE_DELAY
        if isinstance(result, ProcessedPr):
            # Success - add to successful list and update progress
            successful_prs.append(result)
            consecutive_errors = 0  # Reset global error count
            processed_count += 1
            pbar.update(1)
            pbar.set_postfix({
                'success': len(successful_prs),
                'failed': len(failed_prs),
                'retrying': len(active_pr_tasks)
            })

            # Use base delay for successful requests (already set as default)

        elif isinstance(result, FailedPr):
            # Rate limited - calculate delay and retry
            task.retries_remaining -= 1
            if task.retries_remaining <= 0:
                # Max retries exceeded - mark as failed
                failed_prs.append(result)
                processed_count += 1
                pbar.update(1)
                pbar.set_postfix({
                    'success': len(successful_prs),
                    'failed': len(failed_prs),
                    'retrying': len(active_pr_tasks)
                })
                msg = (f"\n  ❌ PR #{task.pr_number} "
                       f"rate limited too many times: ")
                print(msg + f"{result.error_message}")

            else:
                if result.is_rate_limited:
                    consecutive_errors = 0  # Reset global error count
                    # Calculate delay until rate limit reset
                    current_time = datetime.datetime.now(datetime.timezone.utc)
                    reset_time = result.rate_limit_reset_time
                    if reset_time and reset_time > current_time:
                        sleep_time = (reset_time - current_time).total_seconds()
                else:
                    # Fallback to exponential backoff if no reset time
                    sleep_time = BASE_DELAY * (2 ** consecutive_errors)
 
                # Add back to queue for retry
                active_pr_tasks.append(task)
                desc = (f"Processing PRs (rate limited, "
                        f"retrying PR #{task.pr_number})")
                pbar.set_description(desc)
        else:
            raise TypeError(f"Unknown result type: {type(result)}")


        # Unified sleep logic - only sleep if there are more tasks to process
        if active_pr_tasks:
            sleep_time = max(sleep_time, BASE_DELAY)

            if sleep_time > 2:
                # Show detailed timing information
                start_time_str = current_time.strftime('%H:%M:%S UTC')
                print(f"   Wait started: {start_time_str}")
                sleep_delta = datetime.timedelta(seconds=sleep_time)
                print(f"   Wait duration: {sleep_delta}")
                wake_time_str = reset_time.strftime('%H:%M:%S UTC')
                print(f"   Target wake time: {wake_time_str}")
                print("   Press Ctrl-C to abort if needed")
            time.sleep(sleep_time)

    # Close progress bar
    pbar.close()

    # Collect results from successful PR processing
    all_pr_data = []
    for processed_pr in successful_prs:
        pr_data = {
            'pr_number': processed_pr.pr_number,
            'title': processed_pr.title,
            'timestamps': processed_pr.timestamps
        }
        all_pr_data.append(pr_data)

    print("\nPR Processing Summary:")
    print(f"  ✅ Successfully processed: {len(successful_prs)}")
    print(f"  ❌ Failed to process: {len(failed_prs)}")

    if failed_prs:
        print("\nFailed PRs:")
        for failed_pr in failed_prs:
            print(f"  - PR #{failed_pr.pr_number}: {failed_pr.error_message}")

    # Prepare processing statistics
    processing_stats = {
        'total_prs_found': len(filtered_prs),
        'successfully_processed': len(successful_prs),
        'failed_to_process': len(failed_prs),
        'failed_pr_details': [{
            'pr_number': failed_pr.pr_number,
            'error': failed_pr.error_message,
            'was_rate_limited': failed_pr.is_rate_limited
        } for failed_pr in failed_prs]
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
