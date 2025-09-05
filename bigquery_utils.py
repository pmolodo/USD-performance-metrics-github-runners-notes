#!/usr/bin/env python

"""Utilities for working with BigQuery and Google Cloud credentials."""

import argparse
import glob
import os


def setup_credentials(credentials_pattern: str) -> None:
    """
    Set up Google Cloud credentials using a glob pattern.

    Args:
        credentials_pattern: Glob pattern to match credential files

    Raises:
        ValueError: If multiple credential files match the pattern
    """
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
        default=".credentials/*.json",
        help=(
            "Glob pattern for Google Cloud credentials file (default:"
            " '.credentials/*.json')"
        ),
    )
