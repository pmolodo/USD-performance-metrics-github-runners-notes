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
                           output_file: str = "pr_push_times.json"):
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
    """
    # GitHub API base URL
    base_url = 'https://api.github.com'

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

    # Initialize progress bar based on filtered results
    pbar = tqdm(total=len(filtered_prs), desc="Processing PRs", unit="PR")
    all_pr_data = []

    for pr_item in filtered_prs:
        pr_number = pr_item['number']
        pr_created_at = pr_item['created_at']
        pr_title = pr_item['title']
        pr_state = pr_item['state']

        # Update progress bar at start of loop for every PR
        pbar.update(1)

        # Search API already filtered PRs by date, no filtering needed

        # Get detailed PR information to access head/branch info
        pr_details_url = (f'{base_url}/repos/{owner}/{project}/pulls/'
                          f'{pr_number}')
        pr_response = requests.get(pr_details_url, headers=headers,
                                   timeout=30)

        if pr_response.status_code != 200:
            print(f"Warning: Could not fetch PR details for #{pr_number}: "
                  f"{pr_response.status_code}")
            continue

        pr_details = pr_response.json()

        # Get PR branch information
        pr_head_ref = pr_details['head']['ref']  # Branch name
        pr_head_repo = pr_details['head']['repo']['full_name']

        # Get timeline events for this PR to find commit events
        timeline_url = (f'{base_url}/repos/{owner}/{project}/issues/'
                        f'{pr_number}/timeline')
        timeline_response = requests.get(timeline_url, headers=headers,
                                         timeout=30)

        if timeline_response.status_code == 429:
            print(f"    Rate limit hit for PR #{pr_number}, "
                  f"skipping commit events but keeping PR")
            # Still save the PR with just creation time (if within range)
            pr_created_time = parse_datetime_string(
                pr_created_at, is_github_api_format=True)

            timestamps = []
            if is_timestamp_in_range(pr_created_time, start_dt, end_dt):
                timestamps.append(pr_created_at)

            pr_data = {
                'pr_number': pr_number,
                'pr_title': pr_title,
                'pr_created_at': pr_created_at,
                'pr_state': pr_state,
                'pr_branch': f"{pr_head_repo}:{pr_head_ref}",
                'timestamps': timestamps,
                'total_pushes': len(timestamps),
                'note': 'Rate limited - only PR creation time included'
            }
            all_pr_data.append(pr_data)
            continue
        elif timeline_response.status_code != 200:
            print(f"    Warning: Could not fetch timeline for PR "
                  f"#{pr_number}: {timeline_response.status_code}")
            continue

        timeline_events = timeline_response.json()

        # Collect PR creation time + commit event times from timeline
        timestamps = []

        # Filter timeline events for commit events after PR creation
        pr_created_time = parse_datetime_string(
            pr_created_at, is_github_api_format=True)

        # Add PR creation time only if it's within our time range
        if is_timestamp_in_range(pr_created_time, start_dt, end_dt):
            timestamps.append(pr_created_at)

        for event in timeline_events:
            # Look for 'committed' events which indicate commits pushed
            if event.get('event') == 'committed':
                # Get the commit timestamp
                commit_time = event.get('created_at')
                if commit_time:
                    event_time = parse_datetime_string(
                        commit_time, is_github_api_format=True)

                    # Only include commits that are:
                    # 1. After PR creation
                    # 2. Within our specified time range
                    if (event_time > pr_created_time and
                            is_timestamp_in_range(event_time, start_dt,
                                                  end_dt)):
                        timestamps.append(commit_time)

        # Remove duplicates and sort
        unique_timestamps = sorted(list(set(timestamps)))

        pr_data = {
            'pr_number': pr_number,
            'pr_title': pr_title,
            'pr_created_at': pr_created_at,
            'pr_state': pr_state,
            'pr_branch': f"{pr_head_repo}:{pr_head_ref}",
            'timestamps': unique_timestamps,
            'total_pushes': len(unique_timestamps)
        }

        all_pr_data.append(pr_data)

        # Rate limiting - be nice to GitHub API
        time.sleep(0.5)

    # Close progress bar
    pbar.close()

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
        'total_prs': len(all_pr_data),
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
            output_file=args.output
        )
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
