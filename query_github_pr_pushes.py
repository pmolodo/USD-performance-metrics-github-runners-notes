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


def get_total_pr_count(owner: str, project: str, headers: dict) -> int:
    """Get total number of PRs in the repository using search API."""
    try:
        search_url = 'https://api.github.com/search/issues'
        params = {
            'q': f'repo:{owner}/{project} type:pr',
            'per_page': 1  # We only need the count
        }
        response = requests.get(search_url, headers=headers, params=params,
                                timeout=30)
        if response.status_code == 200:
            return response.json().get('total_count', 0)
        else:
            print(f"Warning: Could not get total PR count: "
                  f"{response.status_code}")
            return 0
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"Warning: Could not get total PR count: {e}")
        return 0


def query_github_pr_pushes(owner: str, project: str,
                           start_time: Optional[str] = None,
                           end_time: Optional[str] = None,
                           token: Optional[str] = None,
                           output_file: str = "pr_push_times.json"):
    """
    Query GitHub API for PR creation and commit event timestamps.

    Tracks when PRs are created and when new commits are added to PR branches
    using the timeline API, not individual commit authoring times.

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

    # Get total PR count for progress bar
    total_prs = get_total_pr_count(owner, project, headers)
    if total_prs > 0:
        print(f"Repository has {total_prs} total PRs")
    # Process time filters once outside the loop
    start_dt = None
    end_dt = None
    if start_time:
        start_dt = parse_datetime_string(start_time)
    if end_time:
        end_dt = parse_datetime_string(end_time)

    # Get all pull requests (both open and closed)
    all_pr_data = []
    page = 1
    per_page = 100

    # Initialize progress bar
    pbar = (tqdm(total=total_prs, desc="Processing PRs", unit="PR")
            if total_prs > 0 else None)

    while True:
        url = f'{base_url}/repos/{owner}/{project}/pulls'
        params = {
            'state': 'all',
            'page': page,
            'per_page': per_page,
            'sort': 'created',
            'direction': 'desc'
        }

        response = requests.get(url, headers=headers, params=params,
                                timeout=30)
        if response.status_code == 429:
            print(f"Hit rate limit on page {page}. Saving partial results...")
            break
        elif response.status_code != 200:
            print(f"Error fetching PRs: {response.status_code} - "
                  f"{response.text}")
            if all_pr_data:
                print("Saving partial results before exiting...")
                break
            if pbar:
                pbar.close()
            return

        prs = response.json()
        if not prs:
            break

        # Update progress bar description with current page info
        if pbar:
            pbar.set_description(f"Processing page {page}")
        else:
            print(f"Processing page {page} ({len(prs)} PRs)...")

        for pr in prs:
            pr_number = pr['number']
            pr_created_at = pr['created_at']
            pr_closed_at = pr.get('closed_at')  # None if still open

            # Update progress bar at start of loop for every PR
            if pbar:
                pbar.update(1)

            # Filter PRs by lifetime overlap with requested time range
            if start_dt or end_dt:
                pr_created = parse_datetime_string(
                    pr_created_at, is_github_api_format=True)
                pr_closed = None
                if pr_closed_at:
                    pr_closed = parse_datetime_string(
                        pr_closed_at, is_github_api_format=True)
                else:
                    # If PR is still open, use current time as end
                    pr_closed = datetime.datetime.now(datetime.timezone.utc)

                # Check if PR lifetime overlaps with time range
                # Overlap: PR.created <= script.end AND
                # PR.closed >= script.start
                if start_dt and pr_closed < start_dt:
                    continue
                if end_dt and pr_created > end_dt:
                    continue

            # Only print individual PR progress if no progress bar
            if not pbar:
                print(f"  Processing PR #{pr_number}...")

            # Get PR branch information
            pr_head_ref = pr['head']['ref']  # Branch name
            pr_head_repo = pr['head']['repo']['full_name']

            # Get timeline events for this PR to find commit events
            timeline_url = (f'{base_url}/repos/{owner}/{project}/issues/'
                            f'{pr_number}/timeline')
            timeline_response = requests.get(timeline_url, headers=headers,
                                             timeout=30)

            if timeline_response.status_code == 429:
                print(f"    Rate limit hit for PR #{pr_number}, "
                      f"skipping commit events but keeping PR")
                # Still save the PR with just creation time
                pr_data = {
                    'pr_number': pr_number,
                    'pr_title': pr['title'],
                    'pr_created_at': pr_created_at,
                    'pr_state': pr['state'],
                    'pr_branch': f"{pr_head_repo}:{pr_head_ref}",
                    'timestamps': [pr_created_at],
                    'total_pushes': 1,
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
            timestamps = [pr_created_at]  # Include PR creation time

            # Filter timeline events for commit events after PR creation
            pr_created_time = parse_datetime_string(
                pr_created_at, is_github_api_format=True)

            for event in timeline_events:
                # Look for 'committed' events which indicate commits pushed
                if event.get('event') == 'committed':
                    # Get the commit timestamp
                    commit_time = event.get('created_at')
                    if commit_time:
                        event_time = parse_datetime_string(
                            commit_time, is_github_api_format=True)

                        # Only include commits after PR creation
                        if event_time > pr_created_time:
                            timestamps.append(commit_time)

            # Remove duplicates and sort
            unique_timestamps = sorted(list(set(timestamps)))

            pr_data = {
                'pr_number': pr_number,
                'pr_title': pr['title'],
                'pr_created_at': pr_created_at,
                'pr_state': pr['state'],
                'pr_branch': f"{pr_head_repo}:{pr_head_ref}",
                'timestamps': unique_timestamps,
                'total_pushes': len(unique_timestamps)
            }

            all_pr_data.append(pr_data)

            # Rate limiting - be nice to GitHub API
            time.sleep(0.5)

        page += 1

        # If we got fewer than per_page results, we're done
        if len(prs) < per_page:
            break

    # Close progress bar
    if pbar:
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
