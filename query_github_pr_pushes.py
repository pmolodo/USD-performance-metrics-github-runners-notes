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

###############################################################################
# Core functions
###############################################################################


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

    # Process time filters once outside the loop
    start_dt = None
    end_dt = None
    if start_time:
        start_dt = datetime.datetime.fromisoformat(start_time)
        # Make start_time timezone-aware if it isn't already
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=datetime.timezone.utc)
    if end_time:
        end_dt = datetime.datetime.fromisoformat(end_time)
        # Make end_time timezone-aware if it isn't already
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=datetime.timezone.utc)

    # Get all pull requests (both open and closed)
    all_pr_data = []
    page = 1
    per_page = 100

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
            return

        prs = response.json()
        if not prs:
            break

        print(f"Processing page {page} ({len(prs)} PRs)...")

        for pr in prs:
            pr_number = pr['number']
            pr_created_at = pr['created_at']
            pr_closed_at = pr.get('closed_at')  # None if still open

            # Filter PRs by lifetime overlap with requested time range
            if start_dt or end_dt:
                pr_created = datetime.datetime.fromisoformat(
                    pr_created_at.replace('Z', '+00:00'))
                pr_closed = None
                if pr_closed_at:
                    pr_closed = datetime.datetime.fromisoformat(
                        pr_closed_at.replace('Z', '+00:00'))
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
            pr_created_time = datetime.datetime.fromisoformat(
                pr_created_at.replace('Z', '+00:00'))

            for event in timeline_events:
                # Look for 'committed' events which indicate commits pushed
                if event.get('event') == 'committed':
                    # Get the commit timestamp
                    commit_time = event.get('created_at')
                    if commit_time:
                        event_time = datetime.datetime.fromisoformat(
                            commit_time.replace('Z', '+00:00'))

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

    print(f"\nFound {len(all_pr_data)} PRs with push data.")

    # Save results to JSON file
    output_data = {
        'repository': f'{owner}/{project}',
        'query_timestamp': datetime.datetime.now().isoformat(),
        'filters': {
            'start_time': start_time,
            'end_time': end_time
        },
        'total_prs': len(all_pr_data),
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
