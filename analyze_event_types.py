#!/usr/bin/env python3
"""
Script to analyze all cached timeline query results and extract unique event types.
"""

import json

from collections import Counter, defaultdict
from pathlib import Path


def analyze_cache_event_types(cache_dir: str = ".cache"):
    """
    Analyze all timeline cache files to find unique event types.

    Args:
        cache_dir: Directory containing cache files

    Returns:
        dict: Analysis results with event type counts and examples
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        print(f"Cache directory {cache_dir} does not exist")
        return {}

    # Find all timeline cache files
    timeline_files = list(cache_path.glob("timeline_*.json"))

    if not timeline_files:
        print(f"No timeline cache files found in {cache_dir}")
        return {}

    print(f"Found {len(timeline_files)} timeline cache files")

    # Collect event type statistics
    event_counts = Counter()
    event_examples = defaultdict(list)
    files_processed = 0
    files_with_errors = 0
    total_events = 0

    for cache_file in timeline_files:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # Extract timeline events from response_data
            response_data = cache_data.get("response_data", [])

            if not isinstance(response_data, list):
                print(f"Warning: {cache_file.name} has non-list response_data")
                continue

            files_processed += 1
            file_events = 0

            for event in response_data:
                if not isinstance(event, dict):
                    continue

                event_type = event.get("event")
                if event_type:
                    event_counts[event_type] += 1
                    file_events += 1
                    total_events += 1

                    # Store examples (limit to 3 per event type)
                    if len(event_examples[event_type]) < 3:
                        example = {
                            "file": cache_file.name,
                            "created_at": event.get("created_at"),
                            "actor": event.get("actor", {}).get("login"),
                            "commit_id": event.get("commit_id"),
                        }
                        # Remove None values
                        example = {k: v for k, v in example.items() if v is not None}
                        event_examples[event_type].append(example)

            if file_events > 0:
                print(f"  {cache_file.name}: {file_events} events")

        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"Error processing {cache_file.name}: {e}")
            files_with_errors += 1
            continue

    # Print results
    print("\n=== Event Type Analysis Results ===")
    print(f"Files processed: {files_processed}")
    print(f"Files with errors: {files_with_errors}")
    print(f"Total events analyzed: {total_events}")
    print(f"Unique event types found: {len(event_counts)}")

    if event_counts:
        print("\n=== All Event Types by Frequency (Most Common First) ===")
        for event_type, count in event_counts.most_common():
            percentage = (count / total_events) * 100
            print(f"{event_type:30} {count:6} ({percentage:5.1f}%)")

        print("\n=== Event Type Examples ===")
        for event_type in sorted(event_counts.keys()):
            print(f"\n{event_type}:")
            for example in event_examples[event_type]:
                example_str = ", ".join(f"{k}={v}" for k, v in example.items())
                print(f"  - {example_str}")

    return {
        "event_counts": dict(event_counts),
        "event_examples": dict(event_examples),
        "files_processed": files_processed,
        "files_with_errors": files_with_errors,
        "total_events": total_events,
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze cached timeline query results for event types"
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache",
        help="Directory containing cache files (default: .cache)",
    )
    parser.add_argument("--output", help="Optional JSON file to save detailed results")

    args = parser.parse_args()

    results = analyze_cache_event_types(args.cache_dir)

    if args.output and results:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
