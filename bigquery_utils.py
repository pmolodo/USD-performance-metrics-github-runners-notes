#!/usr/bin/env python

"""Utilities for working with BigQuery and Google Cloud credentials."""

import argparse
import glob
import inspect
import os

# Get the directory where this file is located
THIS_FILE = os.path.abspath(inspect.getsourcefile(lambda: None) or __file__)
THIS_DIR = os.path.dirname(THIS_FILE)

# Default credentials pattern
DEFAULT_CREDENTIALS_PATTERN = ".credentials/*.json"


def setup_credentials(credentials_pattern: str | None) -> None:
    """
    Set up Google Cloud credentials using a glob pattern.

    Args:
        credentials_pattern: Glob pattern to match credential files, or None to
                           use existing credentials or default pattern

    Raises:
        ValueError: If multiple credential files match the pattern
    """
    # If no pattern specified and GOOGLE_APPLICATION_CREDENTIALS is already set, exit early
    if credentials_pattern is None:
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            print("Using existing GOOGLE_APPLICATION_CREDENTIALS environment variable")
            return
        else:
            # Use default pattern relative to THIS_DIR
            credentials_pattern = DEFAULT_CREDENTIALS_PATTERN
            print(
                "No credentials pattern specified, using default:"
                f" {DEFAULT_CREDENTIALS_PATTERN}"
            )

    # If pattern is relative, make it relative to THIS_DIR
    if not os.path.isabs(credentials_pattern):
        credentials_pattern = os.path.join(THIS_DIR, credentials_pattern)

    matches = glob.glob(credentials_pattern)

    if len(matches) == 0:
        # No matches, do nothing (use default credentials)
        print(f"No credential files found matching pattern: {credentials_pattern}")
        print("Using default Google Cloud credentials")
        return
    elif len(matches) == 1:
        # Exactly one match, set environment variable
        credentials_file = os.path.abspath(matches[0])
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file
        print(f"Using credentials file: {credentials_file}")
        return
    else:
        # Multiple matches, error
        raise ValueError(
            "Multiple credential files found matching pattern"
            f" '{credentials_pattern}':\n"
            + "\n".join(f"  - {match}" for match in sorted(matches))
            + "\nPlease specify a more exact glob pattern or filename."
        )


def add_bigquery_args(parser: argparse.ArgumentParser) -> None:
    """
    Add BigQuery-related arguments to an argument parser.

    Args:
        parser: The argument parser to add arguments to
    """
    parser.add_argument(
        "--credentials-file",
        type=str,
        help=(
            "Glob pattern for Google Cloud credentials file. If not specified, will"
            " use existing GOOGLE_APPLICATION_CREDENTIALS or search"
            f" '{DEFAULT_CREDENTIALS_PATTERN}'. Relative patterns are interpreted"
            " relative to the script directory."
        ),
    )
