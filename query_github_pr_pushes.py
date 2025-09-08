#!/usr/bin/env python

"""Query a github project for all times that pushes were made to a PR."""

import argparse
import datetime
import hashlib
import inspect
import json
import os
import sys
import time
import traceback

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import requests

from tqdm import tqdm

# Constants
THIS_FILE = os.path.abspath(inspect.getsourcefile(lambda: None) or __file__)
THIS_DIR = os.path.dirname(THIS_FILE)

DEFAULT_PER_PAGE = 100
BASE_DELAY = 0.5  # seconds
MAX_EXPONENTIAL_DELAY = 300  # seconds
CACHE_VERBOSITY = 2  # Verbosity level required to show cache messages
TIMING_VERBOSITY = 3  # Verbosity level required to show timing information

###############################################################################
# Utility Functions
###############################################################################


def get_current_utc_time() -> datetime.datetime:
    """
    Get the current UTC time.

    Returns:
        Current datetime in UTC timezone
    """
    return datetime.datetime.now(datetime.timezone.utc)


def _get_header_int(headers, header_name: str) -> int | None:
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
        print(f"Warning: Invalid {header_name} header: {header_value!r} ({e})")
        return None


###############################################################################
# Data Structures
###############################################################################


@dataclass
class ApiCallResult:
    """Base class for API call results."""

    api_call_made: bool


@dataclass
class SuccessfulApiCall(ApiCallResult):
    """Result of a successful API call."""

    data: dict


@dataclass
class FailedApiCall(ApiCallResult):
    """Result of a failed API call."""

    error_message: str
    response: object | None = None


@dataclass
class PrResult:
    """Base class for PR processing results."""

    pr_number: int
    api_call_made: bool


@dataclass
class ProcessedPr(PrResult):
    """Result of successfully processing a PR."""

    title: str
    events: list


@dataclass
class FailedPr(PrResult):
    """Result of failed PR processing."""

    error_message: str
    rate_limit_reset_time: datetime.datetime | None = None

    @property
    def is_rate_limited(self) -> bool:
        """True if this failure was due to rate limiting."""
        return self.rate_limit_reset_time is not None

    @classmethod
    def from_response(
        cls, pr_number: int, response, api_call_made: bool = True
    ) -> "FailedPr":
        """
        Create a FailedPr from any non-200 HTTP response.

        Args:
            pr_number: PR number for the FailedPr object
            response: HTTP response object with headers, status_code, text
            api_call_made: Whether an API call was made to get this response

        Returns:
            FailedPr object with appropriate error handling
        """
        # Handle rate limiting (403/429) with special reset time parsing
        if response.status_code in [403, 429]:
            # Try x-ratelimit-reset first (primary rate limits)
            reset_ts = _get_header_int(response.headers, "x-ratelimit-reset")
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
                    api_call_made=api_call_made,
                    error_message=error_message,
                    rate_limit_reset_time=reset_time,
                )

            # Try retry-after (secondary rate limits)
            retry_seconds = _get_header_int(response.headers, "retry-after")
            if retry_seconds is not None:
                current_time = get_current_utc_time()
                delta = datetime.timedelta(seconds=retry_seconds)
                reset_time = current_time + delta
                error_message = (
                    f"Rate limited (retry after {retry_seconds}s, "
                    f"until {reset_time.isoformat()}): "
                    f"{response.status_code} - {response.text}"
                )
                return cls(
                    pr_number=pr_number,
                    api_call_made=api_call_made,
                    error_message=error_message,
                    rate_limit_reset_time=reset_time,
                )

        # Generic HTTP error (including 403/429 without rate limit headers)
        error_message = f"HTTP error {response.status_code}: {response.text}"
        return cls(
            pr_number=pr_number,
            api_call_made=api_call_made,
            error_message=error_message,
        )


@dataclass
class PRTask:
    """Represents a PR processing task with retry information."""

    pr_item: dict
    retries_remaining: int = 3

    @property
    def pr_number(self) -> int:
        """Get PR number from pr_item, raising RuntimeError if missing."""
        number = self.pr_item.get("number")
        if number is None:
            raise RuntimeError("PR item is missing 'number' field")
        return number


###############################################################################
# Cache management functions
###############################################################################


def _build_cache_key(url: str, sorted_params: list | None = None) -> str:
    """
    Build cache key from URL and sorted parameters.

    Args:
        url: The API URL
        sorted_params: Pre-sorted list of (key, value) tuples

    Returns:
        Cache key string for hashing
    """
    cache_key = url
    if sorted_params:
        cache_key += "?" + "&".join(f"{k}={v}" for k, v in sorted_params)
    return cache_key


def get_api_call_type(url: str) -> str:
    """
    Determine API call type from URL.

    Args:
        url: The API URL

    Returns:
        String identifying the API call type

    Raises:
        ValueError: If URL format is not recognized
    """
    if "/timeline" in url:
        return "timeline"
    elif "/events" in url and "/repos/" in url:
        return "repo-events"
    elif "/pulls" in url and "/repos/" in url:
        return "repo-pulls"
    else:
        # Unknown API call - we should handle all specific API types
        raise ValueError(f"Unknown API URL format for caching: {url}")


def get_cache_filename(url: str, params: dict | None = None) -> str:
    """
    Generate a human-readable cache filename based on URL and parameters.

    Args:
        url: The API URL
        params: Optional query parameters

    Returns:
        A human-readable filename safe for filesystem usage
    """
    # Extract API type and relevant info from URL
    api_type = get_api_call_type(url)

    # Extract repo info based on API type
    repo_info = ""
    if api_type == "timeline":
        # Extract repo and PR info from timeline URL
        # Format: /repos/{owner}/{project}/issues/{pr_number}/timeline
        parts = url.split("/")
        if len(parts) >= 7 and "repos" in url:
            owner = parts[parts.index("repos") + 1]
            project = parts[parts.index("repos") + 2]
            if "issues" in parts:
                pr_number = parts[parts.index("issues") + 1]
                repo_info = f"_repo-{owner}-{project}_pr-{pr_number}"
            else:
                repo_info = f"_repo-{owner}-{project}"
    elif api_type == "repo-events":
        # Extract repo info from events URL
        # Format: /repos/{owner}/{project}/events
        parts = url.split("/")
        if len(parts) >= 5 and "repos" in url:
            owner = parts[parts.index("repos") + 1]
            project = parts[parts.index("repos") + 2]
            repo_info = f"_repo-{owner}-{project}"
    elif api_type == "repo-pulls":
        # Extract repo info from pulls URL
        # Format: /repos/{owner}/{project}/pulls
        parts = url.split("/")
        if len(parts) >= 5 and "repos" in url:
            owner = parts[parts.index("repos") + 1]
            project = parts[parts.index("repos") + 2]
            repo_info = f"_repo-{owner}-{project}"

    # Sort params once for both human-readable format and hash
    sorted_params = None
    param_str = ""

    if params:
        sorted_params = sorted(params.items())

        # Format parameters in human-readable way
        param_parts = []
        for key, value in sorted_params:
            # Make parameter names filesystem-safe
            safe_key = str(key).replace("_", "-")
            safe_value = str(value).replace("/", "-").replace(":", "-")
            param_parts.append(f"{safe_key}-{safe_value}")

        if param_parts:
            param_str = "_" + "_".join(param_parts)

    # Create hash for uniqueness using already sorted params
    cache_key = _build_cache_key(url, sorted_params)
    url_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]

    # Create timestamp
    timestamp = get_current_utc_time().isoformat()
    timestamp = timestamp.replace(":", "-").replace(".", "-")

    # Combine all parts: human_readable_hash_timestamp
    filename = f"{api_type}{repo_info}{param_str}_{url_hash}_{timestamp}.json"

    # Ensure filename is not too long (max 255 chars for most filesystems)
    if len(filename) > 200:
        # Truncate param_str if filename is too long
        base_len = len(f"{api_type}{repo_info}_{url_hash}_{timestamp}.json")
        max_param_len = 200 - base_len
        if max_param_len > 0:
            param_str = param_str[:max_param_len]
        filename = f"{api_type}{repo_info}{param_str}_{url_hash}_{timestamp}.json"

    return filename


def ensure_cache_dir() -> Path:
    """
    Ensure the .cache/github_API_calls directory exists and return its path.

    Returns:
        Path object for the .cache/github_API_calls directory
    """
    cache_dir = Path(THIS_DIR) / ".cache" / "github_API_calls"
    cache_dir.mkdir(exist_ok=True, parents=True)
    return cache_dir


def load_cache(
    url: str, params: dict | None = None, max_age_hours: int = 24, verbosity: int = 1
) -> dict | None:
    """
    Load cached API response if it exists and is recent enough.

    Args:
        url: The API URL
        params: Optional query parameters
        max_age_hours: Maximum age of cache in hours (default 24)
        verbosity: Verbosity level for output (default 1)

    Returns:
        Cached response data if found and valid, None otherwise
    """
    cache_dir = ensure_cache_dir()

    # Create hash for exact matching (same logic as get_cache_filename)
    sorted_params = sorted(params.items()) if params else None
    cache_key = _build_cache_key(url, sorted_params)
    url_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]

    # Create pattern to match files with this exact hash
    pattern = f"*_{url_hash}_*.json"

    # Find matching cache files (sorted by modification time, newest first)
    cache_files = sorted(
        cache_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True
    )

    if not cache_files:
        return None

    # Check the most recent cache file
    cache_file = cache_files[0]

    try:
        # Check if cache is too old
        file_mtime = datetime.datetime.fromtimestamp(cache_file.stat().st_mtime)
        cache_age = datetime.datetime.now() - file_mtime
        if cache_age.total_seconds() > max_age_hours * 3600:
            if verbosity >= CACHE_VERBOSITY:
                print(
                    f"Cache file {cache_file.name} is too old ({cache_age}), ignoring"
                )
            return None

        # Load and return cached data
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            if verbosity >= CACHE_VERBOSITY:
                print(f"Using cached response from {cache_file.name}")
            return cached_data

    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not load cache file {cache_file}: {e}")
        return None


def save_cache(
    url: str,
    params: dict | None = None,
    response_data: dict | None = None,
    verbosity: int = 1,
) -> None:
    """
    Save API response to cache.

    Args:
        url: The API URL
        params: Optional query parameters
        response_data: The response data to cache
        verbosity: Verbosity level for output (default 1)
    """
    if response_data is None:
        return

    try:
        cache_dir = ensure_cache_dir()
        filename = get_cache_filename(url, params)
        cache_file = cache_dir / filename

        # Prepare cache data with metadata
        cached_at = get_current_utc_time().isoformat()
        cache_data = {
            "url": url,
            "params": params,
            "cached_at": cached_at,
            "response_data": response_data,
        }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

        if verbosity >= CACHE_VERBOSITY:
            print(f"Saved response to cache: {filename}")

    except OSError as e:
        print(f"Warning: Could not save cache file: {e}")


def sleep_with_timing(
    sleep_seconds: float, verbosity: int = 1, reason: str = "rate limiting"
):
    """
    Sleep for the specified duration with optional timing information.

    Args:
        sleep_seconds: Number of seconds to sleep
        verbosity: Verbosity level for output
        reason: Reason for the sleep (for logging purposes)
    """
    # One-line message if verbosity >= TIMING_VERBOSITY
    if verbosity >= TIMING_VERBOSITY:
        print(f"   Sleeping for {sleep_seconds:.2f}s ({reason})")

    # Detailed timing info if sleep > 2 seconds (always, regardless of
    # verbosity)
    if sleep_seconds > 2:
        current_time = get_current_utc_time()
        start_time_str = current_time.strftime("%H:%M:%S UTC")
        print(f"   Wait started: {start_time_str}")
        sleep_delta = datetime.timedelta(seconds=sleep_seconds)
        print(f"   Wait duration: {sleep_delta}")
        wake_time = current_time + sleep_delta
        wake_time_str = wake_time.strftime("%H:%M:%S UTC")
        print(f"   Target wake time: {wake_time_str}")
        print("   Press Ctrl-C to abort if needed")

    time.sleep(sleep_seconds)


###############################################################################
# API Call Functions
###############################################################################


def cached_api_call(
    url: str,
    headers: dict,
    params: dict | None = None,
    verbosity: int = 1,
    timeout: int = 30,
) -> SuccessfulApiCall | FailedApiCall:
    """
    Make an API call with caching support.

    This function implements the common pattern of:
    1. Check cache first
    2. If not cached, make HTTP request
    3. Handle non-200 responses
    4. Parse JSON response
    5. Save successful response to cache

    Args:
        url: The API URL to call
        headers: HTTP headers for the request
        params: Optional query parameters
        verbosity: Verbosity level for output
        timeout: Request timeout in seconds

    Returns:
        SuccessfulApiCall | FailedApiCall: Object containing the result
    """
    # Check cache first
    cached_data = load_cache(url, params, verbosity=verbosity)
    if cached_data:
        return SuccessfulApiCall(api_call_made=False, data=cached_data["response_data"])

    # Make API request
    try:
        if params:
            response = requests.get(
                url, headers=headers, params=params, timeout=timeout
            )
        else:
            response = requests.get(url, headers=headers, timeout=timeout)

        # Handle non-200 responses
        if response.status_code != 200:
            error_msg = f"API error {response.status_code}: {response.text}"
            return FailedApiCall(
                api_call_made=True, error_message=error_msg, response=response
            )

        # Parse JSON and save to cache
        data = response.json()
        save_cache(url, params, data, verbosity=verbosity)
        return SuccessfulApiCall(api_call_made=True, data=data)

    except requests.RequestException as e:
        error_msg = f"Request failed: {e}"
        return FailedApiCall(api_call_made=False, error_message=error_msg)


###############################################################################
# Core functions
###############################################################################


def filter_events_by_time_range(
    events: list,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
) -> list:
    """
    Filter a list of events by time range.

    Args:
        events: List of events to filter (each event must have 'created_at' field)
        start_time: Start of time range to filter for (inclusive)
        end_time: End of time range to filter for (inclusive)

    Returns:
        list: Filtered events within the specified time range
    """
    if not start_time and not end_time:
        return events

    filtered_events = []
    for event in events:
        event_time = parse_datetime_string(event["created_at"])

        # Filter by time range if specified
        if start_time and event_time < start_time:
            continue
        if end_time and event_time > end_time:
            continue

        filtered_events.append(event)

    return filtered_events


def time_range_overlaps_last_30_days(
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
) -> bool:
    """
    Check if the given time range overlaps with the last 30 days from now.

    Args:
        start_time: Start of time range (None means beginning of time)
        end_time: End of time range (None means current time)

    Returns:
        bool: True if the time range overlaps with the last 30 days
    """
    now = get_current_utc_time()
    thirty_days_ago = now - datetime.timedelta(days=30)

    # Define the last 30 days range: [thirty_days_ago, now]
    last_30_days_start = thirty_days_ago
    last_30_days_end = now

    # Define the requested range, handling None values
    # If start_time is None, it means "beginning of time" (very old date)
    # If end_time is None, it means "current time"
    range_start = (
        start_time
        if start_time is not None
        else datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    )
    range_end = end_time if end_time is not None else now

    # Handle invalid ranges (start_time after end_time)
    if range_start > range_end:
        return False

    # Check if the ranges overlap using standard interval overlap logic:
    # Two intervals [a, b] and [c, d] overlap if max(a, c) <= min(b, d)
    overlap_start = max(last_30_days_start, range_start)
    overlap_end = min(last_30_days_end, range_end)

    return overlap_start <= overlap_end


def fetch_repository_events(
    owner: str,
    project: str,
    headers: dict,
    verbosity: int = 1,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
) -> list:
    """
    Fetch all repository events using a combination of the GitHub Archive and
    the GitHub Events REST API.

    This fetches events like PushEvent, which group commits together naturally,
    providing a more concrete view of push activities compared to individual
    commit timeline events.

    Args:
        owner: Repository owner
        project: Repository name
        headers: HTTP headers for API requests
        verbosity: Verbosity level for output
        start_time: Start of time range to fetch events for
        end_time: End of time range to fetch events for

    Returns:
        List of repository events from both GitHub Archive and REST API
    """

    # Always try to get events from GitHub Archive first

    all_events = fetch_archived_repository_events(
        owner, project, verbosity, start_time, end_time
    )

    if verbosity >= 2:
        print(f"Got {len(all_events)} events from GitHub Archive")

    # If the time range overlaps with the last 30 days, also fetch from REST API
    if time_range_overlaps_last_30_days(start_time, end_time):
        rest_api_events = fetch_repository_events_rest_api(
            owner, project, headers, verbosity, start_time, end_time
        )

        # Merge events, avoiding duplicates based on event ID
        existing_ids = {event.get("id") for event in all_events}
        new_events = [
            event for event in rest_api_events if event.get("id") not in existing_ids
        ]
        all_events.extend(new_events)

        if verbosity >= 2:
            print(
                f"Got {len(rest_api_events)} events from REST API, "
                f"{len(new_events)} were unique"
            )

    if verbosity >= 1:
        print(f"Total events retrieved: {len(all_events)}")

    return all_events


def fetch_archived_repository_events(
    owner: str,
    project: str,
    verbosity: int = 1,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
) -> list:
    """
    Fetch repository events using the GitHub Archive.

    Downloads archive data for the requested time range and filters events
    to match the specified time window.

    Args:
        owner: Repository owner (e.g., "PixarAnimationStudios")
        project: Repository name (e.g., "OpenUSD")
        verbosity: Verbosity level for output
        start_time: Start of time range to fetch events for
        end_time: End of time range to fetch events for

    Returns:
        list: Repository events from GitHub Archive within the time range
    """
    import github_archive_repo_events

    if verbosity >= 1:
        print(f"Fetching repository events from GitHub Archive for {owner}/{project}")

    # Determine month range for months to fetch
    # Use earliest archive date if no start_time specified
    if start_time is None:
        start_month = github_archive_repo_events.EARLIEST_ARCHIVE_MONTH
    else:
        start_month = github_archive_repo_events.Month.from_datetime(start_time)

    # Use current time if no end_time specified
    if end_time is None:
        end_month = github_archive_repo_events.CURRENT_MONTH
    else:
        end_month = github_archive_repo_events.Month.from_datetime(end_time)

    if verbosity >= 2:
        print(f"Time range: {start_time} to {end_time}")
        print(f"Archive months: {start_month} to {end_month}")

    # Use get_repo_events to download and process all events
    all_events = github_archive_repo_events.get_repo_events(
        repo_owner=owner,
        repo_name=project,
        start_month=start_month,
        end_month=end_month,
    )

    # Filter events by time range using common filtering function
    filtered_events = filter_events_by_time_range(all_events, start_time, end_time)

    if verbosity >= 1:
        print(f"Fetched {len(filtered_events)} events from GitHub Archive")

    return filtered_events


def fetch_repository_events_rest_api(
    owner: str,
    project: str,
    headers: dict,
    verbosity: int = 1,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
) -> list:
    """
    Fetch all repository events using the GitHub Events API.

    This fetches events like PushEvent, which group commits together naturally,
    providing a more concrete view of push activities compared to individual
    commit timeline events.

    Args:
        owner: Repository owner
        project: Repository name
        headers: HTTP headers for API requests
        verbosity: Verbosity level for output
        start_time: Start of time range to fetch events for
        end_time: End of time range to fetch events for

    Returns:
        List of repository events from the GitHub Events API within the time range
    """
    events_url = f"https://api.github.com/repos/{owner}/{project}/events"
    all_events = []
    page = 1

    if verbosity >= 1:
        print(f"Fetching repository events from {owner}/{project}...")

    # Initialize progress bar for events pagination
    pbar = tqdm(desc="Fetching repository events", unit="event")

    while True:
        params = {
            "page": page,
            "per_page": DEFAULT_PER_PAGE,
        }

        # Make API call with caching
        api_result = cached_api_call(events_url, headers, params, verbosity=verbosity)

        if isinstance(api_result, FailedApiCall):
            print(f"Error fetching repository events: {api_result.error_message}")
            break

        events_data = api_result.data

        if not events_data:
            break

        all_events.extend(events_data)

        # Update progress bar
        pbar.n = len(all_events)
        pbar.set_description(f"Fetching repository events ({len(all_events)} total)")
        pbar.refresh()

        # GitHub's events API returns up to 300 events and only includes
        # events from the past 30 days, so we don't need to worry about
        # infinite pagination
        if len(events_data) < DEFAULT_PER_PAGE:
            break

        page += 1

    # Final update to show completion
    pbar.n = len(all_events)
    pbar.set_description(f"Fetched {len(all_events)} repository events")
    pbar.refresh()
    pbar.close()

    if verbosity >= 1:
        print(f"Retrieved {len(all_events)} repository events from REST API")

    # Filter events by time range using common filtering function
    filtered_events = filter_events_by_time_range(all_events, start_time, end_time)

    if verbosity >= 1 and len(filtered_events) != len(all_events):
        print(f"Filtered to {len(filtered_events)} events within time range")

    return filtered_events


def get_repository_pull_requests(
    owner: str,
    project: str,
    headers: dict,
    state: str = "all",
    head: str | None = None,
    base: str | None = None,
    sort: str = "created",
    direction: str = "desc",
    per_page: int | None = None,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
    max_prs: int | None = None,
    verbosity: int = 1,
) -> list:
    """
    Fetch all pull requests for a repository using GitHub's REST API.

    Uses the /repos/{owner}/{repo}/pulls endpoint to retrieve pull requests
    with support for filtering by state, branches, sorting options, and time ranges.

    Args:
        owner: Repository owner
        project: Repository name
        headers: HTTP headers for API requests
        state: The state of the pull requests. Can be either 'open', 'closed', or 'all'
        head: Filter pulls by head user or head organization and branch name in the format
              'user:ref-name' or 'organization:ref-name'
        base: Filter pulls by base branch name
        sort: What to sort results by. Can be either 'created', 'updated', 'popularity'
              (comment count) or 'long-running' (age, filtering by pulls updated in the last month)
        direction: The direction of the sort. Can be either 'asc' or 'desc'
        per_page: Results per page (max 100), defaults to DEFAULT_PER_PAGE
        start_time: Optional start time filter (filters by created_at field)
        end_time: Optional end time filter (filters by created_at field)
        max_prs: Optional maximum number of PRs to return
        verbosity: Verbosity level for output

    Returns:
        List of pull request objects from the GitHub API, filtered by time range if specified
    """
    # Validate max_prs is an int >= 0. If 0, exit immediately.
    if max_prs is not None:
        if not isinstance(max_prs, int) or max_prs < 0:
            raise ValueError("max_prs must be an integer >= 0")
        if max_prs == 0:
            return []

    if per_page is None:
        per_page = DEFAULT_PER_PAGE
    pulls_url = f"https://api.github.com/repos/{owner}/{project}/pulls"
    all_pulls = []
    page = 1

    if verbosity >= 1:
        print(f"Fetching pull requests from {owner}/{project}...")

    # Initialize progress bar for pulls pagination
    pbar = tqdm(desc="Fetching pull requests", unit="pull")

    while True:
        # Calculate per_page and remaining_results only if max_prs is set
        if max_prs is None:
            current_per_page = per_page
        else:
            remaining_results = max_prs - len(all_pulls)
            current_per_page = min(remaining_results, per_page)

        params = {
            "state": state,
            "sort": sort,
            "direction": direction,
            "page": page,
            "per_page": current_per_page,
        }

        # Add optional parameters if provided
        if head is not None:
            params["head"] = head
        if base is not None:
            params["base"] = base

        # Make API call with caching
        api_result = cached_api_call(pulls_url, headers, params, verbosity=verbosity)

        if isinstance(api_result, FailedApiCall):
            print(f"Error fetching pull requests: {api_result.error_message}")
            break

        pulls_data = api_result.data

        if not pulls_data:
            break

        # Filter pulls by time range if specified
        if start_time is not None or end_time is not None:
            filtered_pulls = []
            for pull in pulls_data:
                created_at_str = pull.get("created_at")
                if created_at_str:
                    created_at = parse_datetime_string(created_at_str)
                    if is_timestamp_in_range(created_at, start_time, end_time):
                        filtered_pulls.append(pull)
            pulls_data = filtered_pulls

        all_pulls.extend(pulls_data)

        # Update progress bar
        pbar.n = len(all_pulls)
        desc = f"Fetching pull requests ({len(all_pulls)} total"
        if max_prs is not None:
            desc += f"/{max_prs}"
        desc += ")"
        pbar.set_description(desc)
        pbar.refresh()

        # Check if we've reached max_prs limit
        if max_prs is not None and len(all_pulls) >= max_prs:
            all_pulls = all_pulls[:max_prs]
            break

        # Check if we've reached the end of results
        if len(pulls_data) < current_per_page:
            break

        page += 1

    # Final update to show completion
    pbar.n = len(all_pulls)
    pbar.set_description(f"Fetched {len(all_pulls)} pull requests")
    pbar.refresh()
    pbar.close()

    if verbosity >= 1:
        print(f"Retrieved {len(all_pulls)} pull requests from REST API")

    return all_pulls


def group_prs_by_head_ref(filtered_prs: list) -> dict:
    """
    Group pull requests by their head repository and branch reference.

    Args:
        filtered_prs: List of PR objects from get_repository_pull_requests

    Returns:
        Dict with three-level nesting: owner -> repo -> ref -> list of PR numbers.
        Structure: {
            "owner1": {
                "repo1": {
                    "branch_name": [pr_number1, pr_number2, ...]
                }
            }
        }

        For PRs from forks (where head.repo is null), uses the fork owner's username
        as the owner and "fork" as the repository name.
    """
    owner_repo_ref_map = {}

    for pr in filtered_prs:
        pr_number = pr.get("number")
        if pr_number is None:
            raise ValueError(
                "PR missing required 'number' field. This indicates corrupted GitHub"
                " API data."
            )

        head_info = pr.get("head")
        if not head_info:
            raise ValueError(
                f"PR #{pr_number}: missing required 'head' field. This indicates"
                " corrupted GitHub API data."
            )

        ref = head_info.get("ref")
        if not ref:
            raise ValueError(
                f"PR #{pr_number}: missing required 'head.ref' field. This indicates"
                " corrupted GitHub API data."
            )

        # Determine owner and repository
        head_repo = head_info.get("repo")
        if head_repo is None:
            # Skip PRs with null repo (deleted or private fork repositories)
            head_user = head_info.get("user")
            user_login = head_user.get("login") if head_user else "unknown"
            print(
                f"Warning: Skipping PR #{pr_number} - fork repository from user"
                f" '{user_login}' is deleted or private"
            )
            continue

        # If repo exists, it should always have full_name
        full_name = head_repo.get("full_name")
        if not full_name:
            raise ValueError(
                f"PR #{pr_number}: repository exists but missing full_name. "
                "This indicates corrupted or unexpected GitHub API data."
            )

        # Parse full repository name: "owner/repo"
        if "/" in full_name:
            owner, repo = full_name.split("/", 1)
        else:
            raise ValueError(
                f"PR #{pr_number}: repository full_name '{full_name}' does not contain"
                " '/'. Expected format is 'owner/repo'. This indicates unexpected"
                " GitHub API data."
            )

        # Initialize three-level nested dict structure if needed
        repo_ref_map = owner_repo_ref_map.setdefault(owner, {})
        ref_map = repo_ref_map.setdefault(repo, {})
        pr_list = ref_map.setdefault(ref, [])

        # Add PR number to the list
        pr_list.append(pr_number)

    return owner_repo_ref_map


def get_pr_push_events(
    repo_refs: dict[str, dict[str, Iterable[str]]],
    headers: dict,
    verbosity: int = 1,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
) -> dict:
    """
    Fetch push events for all repositories and filter to only include specified refs.

    Args:
        repo_refs: Two-level nested dict: owner -> repo -> iterable of ref names
        headers: HTTP headers for API requests
        verbosity: Verbosity level for output
        start_time: Start of time range to fetch events for
        end_time: End of time range to fetch events for

    Returns:
        Three-level nested dict: owner -> repo -> ref -> list of PushEvent objects,
        filtered to only include refs that appear in the input mapping
    """
    aggregated_push_events = {}
    total_repos = 0
    total_events = 0

    for owner, repos in repo_refs.items():
        repo_events = aggregated_push_events.setdefault(owner, {})

        for repo, ref_names in repos.items():
            ref_events = repo_events.setdefault(repo, {})
            repo_id = f"{owner}/{repo}"

            # Convert ref names to set for efficient lookups
            target_refs = set(ref_names)

            if verbosity >= 2:
                print(f"  Processing {repo_id} with {len(target_refs)} refs...")

            # Fetch push events for this repository
            repo_push_events = get_repository_push_events(
                owner,
                repo,
                headers,
                verbosity=max(0, verbosity - 1),
                start_time=start_time,
                end_time=end_time,
            )

            # Process each requested ref to ensure all appear in output
            for ref_name in ref_names:
                # Convert short ref name to full format for lookup
                full_ref = f"refs/heads/{ref_name}"
                events = repo_push_events.get(full_ref, [])

                # Use short ref name in output to match input format
                ref_events[ref_name] = events
                total_events += len(events)

                if verbosity >= 3:
                    print(f"    Added {len(events)} push events for ref {ref_name}")

            total_repos += 1

    if verbosity >= 1:
        print(f"Aggregated {total_events} push events from {total_repos} repositories")

    return aggregated_push_events


def get_repository_push_events(
    owner: str,
    project: str,
    headers: dict,
    verbosity: int = 1,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
) -> dict:
    """
    Fetch repository events and extract PushEvents organized by ref.

    Args:
        owner: Repository owner
        project: Repository name
        headers: HTTP headers for API requests
        verbosity: Verbosity level for output
        start_time: Start of time range to fetch events for
        end_time: End of time range to fetch events for

    Returns:
        Dictionary mapping git refs to lists of PushEvent objects
    """
    # Fetch repository events first
    repository_events = fetch_repository_events(
        owner, project, headers, verbosity, start_time, end_time
    )

    # Process repository events to extract PushEvents organized by ref
    push_events_by_ref = {}
    push_event_count = 0

    for event in repository_events:
        if event.get("type") == "PushEvent":
            push_event_count += 1
            payload = event.get("payload", {})
            ref = payload.get("ref")

            if ref:
                if ref not in push_events_by_ref:
                    push_events_by_ref[ref] = []
                push_events_by_ref[ref].append(event)

    if verbosity >= 1:
        print(
            f"Found {push_event_count} PushEvents across {len(push_events_by_ref)} refs"
        )
        if verbosity >= 2:
            for ref, events in push_events_by_ref.items():
                print(f"  {ref}: {len(events)} push events")

    return push_events_by_ref


def parse_datetime_string(dt_string: str) -> datetime.datetime:
    """
    Convert a datetime string to a timezone-aware datetime object.

    Args:
        dt_string: The datetime string to parse
        is_github_api_format: If True, converts 'Z' suffix to '+00:00'
                             If False, assumes user input and adds UTC timezone

    Returns:
        A timezone-aware datetime object
    """
    # GitHub API uses 'Z' suffix which needs to be converted to '+00:00'
    if dt_string.endswith("Z"):
        dt_string = dt_string[:-1] + "+00:00"

    dt = datetime.datetime.fromisoformat(dt_string)

    # Ensure timezone awareness - add UTC if no timezone specified
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)

    return dt


def is_timestamp_in_range(
    timestamp: datetime.datetime,
    start_dt: datetime.datetime | None,
    end_dt: datetime.datetime | None,
) -> bool:
    """
    Check if a timestamp falls within the specified time range.

    Args:
        timestamp: The datetime to check
        start_dt: Optional start datetime (None means no start limit)
        end_dt: Optional end datetime (None means no end limit)

    Returns:
        True if timestamp is within range, False otherwise
    """
    return (start_dt is None or timestamp >= start_dt) and (
        end_dt is None or timestamp <= end_dt
    )


def get_event_time_str(event: dict) -> str:
    """Get the time string from a timeline event."""
    event_time = event.get("created_at")
    if event_time is not None:
        return event_time

    raise ValueError(f"No time string found for event: {event}")


def get_event_time(event: dict) -> datetime.datetime:
    """Get the datetime object from a timeline event."""
    return parse_datetime_string(get_event_time_str(event))


def get_pr_push_events_list(pr_item: dict, push_events_by_ref: dict) -> list:
    """
    Extract push events for a specific PR from the structured push events data.

    Args:
        pr_item: PR object from GitHub API (assumes repo exists - nulls filtered earlier)
        push_events_by_ref: Three-level nested dict: owner -> repo -> ref -> PushEvent lists

    Returns:
        List of push events for this PR's head ref, or empty list if not found
    """
    head_info = pr_item["head"]
    ref = head_info["ref"]

    # Extract head repository owner and name (repo guaranteed to exist)
    repo_info = head_info["repo"]
    owner, repo_name = repo_info["full_name"].split("/", 1)

    # Navigate the 3-level structure to get the events
    return push_events_by_ref.get(owner, {}).get(repo_name, {}).get(ref, [])


def process_single_pr(
    task: PRTask,
    headers: dict,
    owner: str,
    project: str,
    start_dt: datetime.datetime | None,
    end_dt: datetime.datetime | None,
    push_events: list,
    verbosity: int = 1,
) -> ProcessedPr | FailedPr:
    """Process a single PR to extract timeline data.

    Args:
        task: The PR task containing data to process
        headers: HTTP headers for API requests
        owner: Repository owner
        project: Repository name
        start_dt: Optional start datetime filter
        end_dt: Optional end datetime filter
        push_events: List of push events for this PR's head ref
        verbosity: Verbosity level for output

    Returns:
        ProcessedPr on success, FailedPr on failure
    """
    pr_item = task.pr_item
    pr_number = task.pr_number

    try:
        # Get PR creation time (always included)
        pr_created_str = pr_item["created_at"]

        pr_created = parse_datetime_string(pr_created_str)

        # Start with PR creation timestamp
        events = []
        if is_timestamp_in_range(pr_created, start_dt, end_dt):
            events.append({"event": "created", "time": pr_created.isoformat()})

        # Get timeline events
        timeline_url = (
            "https://api.github.com/repos/"
            f"{owner}/{project}/"
            f"issues/{pr_number}/timeline"
        )

        # Make API call with caching
        api_result = cached_api_call(timeline_url, headers, verbosity=verbosity)

        if isinstance(api_result, FailedApiCall):
            # For timeline API, we need to return a proper FailedPr
            if api_result.response is not None:
                # We have a real response object from the failed request
                return FailedPr.from_response(
                    pr_number, api_result.response, api_result.api_call_made
                )
            else:
                # Network/request error - create a basic FailedPr
                return FailedPr(
                    pr_number=pr_number,
                    error_message=f"Request failed: {api_result.error_message}",
                    api_call_made=api_result.api_call_made,
                )

        # Process timeline events for commit-related activity
        timeline_events = api_result.data
        if timeline_events:  # Ensure timeline_events is not None
            for event in timeline_events:
                event_type = event["event"]
                if event_type in [
                    "head_ref_force_pushed",
                    "head_ref_restored",
                    "merged",
                    "reopened",
                    "closed",
                ]:
                    event_time = get_event_time(event)
                    if not is_timestamp_in_range(event_time, start_dt, end_dt):
                        continue

                    event_type = event["event"]
                    commit_id = event.get("commit_id")

                    if event_type == "closed" and commit_id is None:
                        # We're only interested in "closed" events that have an
                        # associated commit ID, which means that they were
                        # closed because a commit has a "closes" or "fixes"
                        # comment - which we consider similar to a merge.
                        continue

                    if not commit_id:
                        commit_id = event.get("sha")

                    event_data = {"event": event_type, "time": event_time.isoformat()}
                    if commit_id is not None:
                        event_data["commit_id"] = commit_id

                    events.append(event_data)

        # Process push events for this PR
        for push_event in push_events:
            push_time_str = push_event["created_at"]
            push_time = parse_datetime_string(push_time_str)
            if is_timestamp_in_range(push_time, start_dt, end_dt):
                push_data = {
                    "event": "push",
                    "time": push_time.isoformat(),
                }

                # Add commit info if available
                payload = push_event["payload"]
                commits = payload.get("commits", [])
                if commits:
                    push_data["commit_count"] = str(len(commits))
                    # Add the head commit sha
                    head_commit = payload.get("head")
                    if head_commit:
                        push_data["commit_id"] = head_commit

                events.append(push_data)

        # Return successful result
        return ProcessedPr(
            pr_number=pr_number,
            title=pr_item.get("title", ""),
            events=events,
            api_call_made=api_result.api_call_made,
        )

    except requests.RequestException as e:
        return FailedPr(
            pr_number=pr_number,
            error_message=f"Request failed: {e}",
            api_call_made=False,
        )


def query_github_pr_pushes(
    owner: str,
    project: str,
    start_time: str | None = None,
    end_time: str | None = None,
    token: str | None = None,
    output_file: str = "pr_push_times.json",
    max_prs: int | None = None,
    verbosity: int = 1,
):
    """
    Query GitHub API for PR creation and commit events.

    Tracks when PRs are created and when new commits are added to PR branches
    using the timeline API, not individual commit authoring times.

    Filters PRs by lifetime overlap with time range, and filters all collected
    events to only include those within the specified time range.

    Args:
        owner: Repository owner
        project: Repository name
        start_time: Optional start time filter (ISO format string)
        end_time: Optional end time filter (ISO format string)
        token: GitHub personal access token
        output_file: Output JSON file path
        max_prs: Optional maximum number of PRs to process (for testing)
        verbosity: Verbosity level for output (default 1)
    """
    # Setup headers
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "USD-Performance-Metrics-Script",
    }

    # Add authentication if token provided
    if token:
        headers["Authorization"] = f"token {token}"
    elif "GITHUB_TOKEN" in os.environ:
        headers["Authorization"] = f'token {os.environ["GITHUB_TOKEN"]}'
    else:
        print("Warning: No GitHub token provided. You may hit rate limits.")
        print(
            "Consider using --token YOUR_TOKEN or setting GITHUB_TOKEN "
            "environment variable."
        )

    print(f"Fetching PRs for {owner}/{project}...")

    # Process time filters once outside the loop
    start_dt = None
    end_dt = None
    if start_time:
        start_dt = parse_datetime_string(start_time)
    if end_time:
        end_dt = parse_datetime_string(end_time)

    print("Gathering PRs...")
    filtered_prs = get_repository_pull_requests(
        owner=owner,
        project=project,
        headers=headers,
        start_time=start_dt,
        end_time=end_dt,
        max_prs=max_prs,
        verbosity=verbosity,
    )

    if not filtered_prs:
        print("No PRs found matching the specified criteria.")
        return

    prs_by_repo_ref = group_prs_by_head_ref(filtered_prs)

    # Fetch and process repository events to extract PushEvents organized by ref
    push_events_by_ref = get_pr_push_events(
        prs_by_repo_ref, headers, verbosity, start_dt, end_dt
    )

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

        # Get push events for this specific PR
        pr_push_events = get_pr_push_events_list(task.pr_item, push_events_by_ref)

        # Process the PR and get result
        result = process_single_pr(
            task,
            headers,
            owner,
            project,
            start_dt,
            end_dt,
            pr_push_events,
            verbosity,
        )

        # Initialize sleep time for this iteration
        sleep_time = BASE_DELAY
        if isinstance(result, ProcessedPr):
            # Success - add to successful list and update progress
            successful_prs.append(result)
            consecutive_errors = 0  # Reset global error count
            processed_count += 1
            pbar.update(1)
            pbar.set_postfix(
                {
                    "success": len(successful_prs),
                    "failed": len(failed_prs),
                    "retrying": len(active_pr_tasks),
                }
            )

            # Use base delay for successful requests (already set as default)

        elif isinstance(result, FailedPr):
            # Rate limited - calculate delay and retry
            task.retries_remaining -= 1
            if task.retries_remaining <= 0:
                # Max retries exceeded - mark as failed
                failed_prs.append(result)
                processed_count += 1
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "success": len(successful_prs),
                        "failed": len(failed_prs),
                        "retrying": len(active_pr_tasks),
                    }
                )
                msg = f"\n  ❌ PR #{task.pr_number} rate limited too many times: "
                print(msg + f"{result.error_message}")

            else:
                if result.is_rate_limited:
                    consecutive_errors = 0  # Reset global error count
                    # Calculate delay until rate limit reset
                    current_time = get_current_utc_time()
                    reset_time = result.rate_limit_reset_time
                    if reset_time and reset_time > current_time:
                        delta = reset_time - current_time
                        sleep_time = delta.total_seconds()
                else:
                    # Fallback to exponential backoff if no reset time
                    sleep_time = BASE_DELAY * (2**consecutive_errors)

                # Add back to queue for retry
                active_pr_tasks.append(task)
                desc = f"Processing PRs (rate limited, retrying PR #{task.pr_number})"
                pbar.set_description(desc)
        else:
            raise TypeError(f"Unknown result type: {type(result)}")

        # Unified sleep logic - only sleep if there are more tasks to process
        # and an API call was made (rate limiting applies)
        if active_pr_tasks and result.api_call_made:
            sleep_time = max(sleep_time, BASE_DELAY)

            # Determine reason for sleep based on context
            if isinstance(result, FailedPr) and result.is_rate_limited:
                reason = "rate limit reset"
            elif isinstance(result, FailedPr):
                reason = "exponential backoff"
            else:
                reason = "rate limiting prevention"

            sleep_with_timing(sleep_time, verbosity, reason)

    # Close progress bar
    pbar.close()

    # Collect results from successful PR processing
    all_pr_data = []
    for processed_pr in successful_prs:
        pr_data = {
            "pr_number": processed_pr.pr_number,
            "title": processed_pr.title,
            "events": processed_pr.events,
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
        "total_prs_found": len(filtered_prs),
        "successfully_processed": len(successful_prs),
        "failed_to_process": len(failed_prs),
        "failed_pr_details": [
            {
                "pr_number": failed_pr.pr_number,
                "error": failed_pr.error_message,
                "was_rate_limited": failed_pr.is_rate_limited,
            }
            for failed_pr in failed_prs
        ],
    }

    # Calculate total events across all PRs
    total_events = sum(len(pr_data["events"]) for pr_data in all_pr_data)

    print(f"\nFound {len(all_pr_data)} PRs with push data.")
    print(f"Total events across all PRs: {total_events}")

    # Save results to JSON file
    output_data = {
        "repository": f"{owner}/{project}",
        "query_timestamp": datetime.datetime.now().isoformat(),
        "filters": {"start_time": start_time, "end_time": end_time},
        "processing_statistics": processing_stats,
        "total_prs_processed": len(all_pr_data),
        "total_events": total_events,
        "prs": all_pr_data,
    }

    with open(output_file, "w", encoding="utf-8") as f:
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
    parser.add_argument(
        "--start",
        help="Optional start time filter (ISO format, e.g., 2023-01-01T00:00:00)",
    )
    parser.add_argument(
        "--end",
        help="Optional end time filter (ISO format, e.g., 2023-12-31T23:59:59)",
    )
    parser.add_argument(
        "--token",
        help="GitHub personal access token (can also use GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--output", default="pr_push_times.json", help="Output JSON file path"
    )
    parser.add_argument(
        "--max-prs", type=int, help="Maximum number of PRs to process (for testing)"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity level (use -v, -vv, -vvv, etc.)",
    )
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
            max_prs=args.max_prs,
            verbosity=args.verbose,
        )
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
