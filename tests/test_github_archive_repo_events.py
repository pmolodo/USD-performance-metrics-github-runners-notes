#!/usr/bin/env python

"""Tests for github_archive_repo_events module."""

import os
import sys

import pytest

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import github_archive_repo_events


class TestRecursiveDictMerge:
    """Tests for recursive_dict_merge function."""

    def test_basic_merge_no_conflicts(self):
        """Test basic merging with no overlapping keys."""
        target = {"a": 1, "b": 2}
        source = {"c": 3, "d": 4}

        github_archive_repo_events.recursive_dict_merge(target, source)

        expected = {"a": 1, "b": 2, "c": 3, "d": 4}
        assert target == expected

    def test_recursive_merge_nested_dicts(self):
        """Test recursive merging of nested dictionaries."""
        target = {
            "level1": {"level2a": {"x": 10}, "level2b": {"p": 100}},
            "other": "value",
        }
        source = {
            "level1": {"level2a": {"z": 30}, "level2c": {"q": 200}},
            "new": "data",
        }

        github_archive_repo_events.recursive_dict_merge(target, source)

        expected = {
            "level1": {
                "level2a": {"x": 10, "z": 30},
                "level2b": {"p": 100},
                "level2c": {"q": 200},
            },
            "other": "value",
            "new": "data",
        }
        assert target == expected

    def test_deep_nested_merge(self):
        """Test merging with deeply nested structures."""
        target = {"a": {"b": {"c": {"d": {"existing": 1}}}}}
        source = {"a": {"b": {"c": {"d": {"new": 2}, "e": {"other": 3}}}}}

        github_archive_repo_events.recursive_dict_merge(target, source)

        expected = {
            "a": {"b": {"c": {"d": {"existing": 1, "new": 2}, "e": {"other": 3}}}}
        }
        assert target == expected

    def test_empty_source_dict(self):
        """Test merging with empty source dictionary."""
        target = {"a": 1, "b": 2}
        source = {}
        original_target = target.copy()

        github_archive_repo_events.recursive_dict_merge(target, source)

        assert target == original_target

    def test_empty_target_dict(self):
        """Test merging into empty target dictionary."""
        target = {}
        source = {"a": 1, "b": 2}

        github_archive_repo_events.recursive_dict_merge(target, source)

        assert target == source

    def test_both_empty_dicts(self):
        """Test merging two empty dictionaries."""
        target = {}
        source = {}

        github_archive_repo_events.recursive_dict_merge(target, source)

        assert target == {}

    def test_conflict_string_vs_dict_raises_error(self):
        """Test that conflicting string vs dict raises ValueError."""
        target = {"key": "string_value"}
        source = {"key": {"nested": "dict"}}

        with pytest.raises(ValueError) as exc_info:
            github_archive_repo_events.recursive_dict_merge(target, source)

        error_msg = str(exc_info.value)
        assert "Key 'key' exists in both dictionaries" in error_msg
        assert "Target type: <class 'str'>" in error_msg
        assert "Source type: <class 'dict'>" in error_msg

    def test_conflict_dict_vs_string_raises_error(self):
        """Test that conflicting dict vs string raises ValueError."""
        target = {"key": {"nested": "dict"}}
        source = {"key": "string_value"}

        with pytest.raises(ValueError) as exc_info:
            github_archive_repo_events.recursive_dict_merge(target, source)

        error_msg = str(exc_info.value)
        assert "Key 'key' exists in both dictionaries" in error_msg
        assert "Target type: <class 'dict'>" in error_msg
        assert "Source type: <class 'str'>" in error_msg

    def test_conflict_int_vs_list_raises_error(self):
        """Test that conflicting int vs list raises ValueError."""
        target = {"key": 42}
        source = {"key": [1, 2, 3]}

        with pytest.raises(ValueError) as exc_info:
            github_archive_repo_events.recursive_dict_merge(target, source)

        error_msg = str(exc_info.value)
        assert "Key 'key' exists in both dictionaries" in error_msg
        assert "Target type: <class 'int'>" in error_msg
        assert "Source type: <class 'list'>" in error_msg

    def test_conflict_same_type_non_dict_raises_error(self):
        """Test that conflicting values of same non-dict type raise ValueError."""
        target = {"key": "old_value"}
        source = {"key": "new_value"}

        with pytest.raises(ValueError) as exc_info:
            github_archive_repo_events.recursive_dict_merge(target, source)

        error_msg = str(exc_info.value)
        assert "Key 'key' exists in both dictionaries" in error_msg
        assert "Target type: <class 'str'>" in error_msg
        assert "Source type: <class 'str'>" in error_msg

    def test_equal_values_no_error(self):
        """Test that equal values don't raise error even when not both dicts."""
        target = {
            "string_key": "same_value",
            "int_key": 42,
            "list_key": [1, 2, 3],
            "none_key": None,
            "nested": {"inner_key": "inner_value"},
        }
        source = {
            "string_key": "same_value",  # Same string
            "int_key": 42,  # Same int
            "list_key": [1, 2, 3],  # Same list
            "none_key": None,  # Same None
            "new_key": "added_value",  # New key
            "nested": {
                "inner_key": "inner_value",  # Same nested value
                "new_inner": "new_value",  # New nested key
            },
        }

        # This should not raise an error
        github_archive_repo_events.recursive_dict_merge(target, source)

        expected = {
            "string_key": "same_value",
            "int_key": 42,
            "list_key": [1, 2, 3],
            "none_key": None,
            "new_key": "added_value",
            "nested": {"inner_key": "inner_value", "new_inner": "new_value"},
        }
        assert target == expected

    def test_conflict_nested_path_raises_error(self):
        """Test that conflicts in nested paths raise appropriate errors."""
        target = {"level1": {"level2": {"conflict_key": "string"}}}
        source = {"level1": {"level2": {"conflict_key": {"nested": "dict"}}}}

        with pytest.raises(ValueError) as exc_info:
            github_archive_repo_events.recursive_dict_merge(target, source)

        error_msg = str(exc_info.value)
        assert "Key 'conflict_key' exists in both dictionaries" in error_msg

    # Current behavior, but not desired - don't create test
    # def test_partial_merge_before_error(self):
    #     """Test that target is partially modified before error occurs."""
    #     target = {
    #         "safe_key": "safe_value",
    #         "nested": {"safe_nested": "safe", "conflict_key": "string"},
    #     }
    #     source = {
    #         "new_key": "new_value",
    #         "nested": {"new_nested": "new", "conflict_key": {"nested": "dict"}},
    #     }

    #     with pytest.raises(ValueError):
    #         github_archive_repo_events.recursive_dict_merge(target, source)

    #     # Target should be partially modified (new_key added, safe merges done)
    #     # but the conflict should have prevented full completion
    #     assert "new_key" in target
    #     assert target["new_key"] == "new_value"
    #     assert target["nested"]["safe_nested"] == "safe"  # Original preserved
    #     assert "new_nested" in target["nested"]  # New key added
    #     assert target["nested"]["new_nested"] == "new"

    def test_complex_real_world_scenario(self):
        """Test a complex scenario similar to GitHub event merging."""
        target = {
            "type": "PushEvent",
            "actor": {"id": 123, "login": "testuser"},
            "repo": {"id": 456, "name": "test/repo"},
        }
        source = {
            "action": "created",
            "actor": {
                "display_login": "Test User",
                "avatar_url": "https://example.com/avatar.png",
            },
            "additional_info": {"timestamp": "2025-01-01T00:00:00Z"},
        }

        github_archive_repo_events.recursive_dict_merge(target, source)

        expected = {
            "type": "PushEvent",
            "action": "created",
            "actor": {
                "id": 123,
                "login": "testuser",
                "display_login": "Test User",
                "avatar_url": "https://example.com/avatar.png",
            },
            "repo": {"id": 456, "name": "test/repo"},
            "additional_info": {"timestamp": "2025-01-01T00:00:00Z"},
        }
        assert target == expected

    def test_none_values_handled_correctly(self):
        """Test that None values are handled correctly."""
        target = {"key1": None, "key2": {"nested": None}}
        source = {"key1": "value", "key2": {"nested": "value", "new": "data"}}

        # This should raise an error because key1 has None (not dict) in target
        # and string (not dict) in source - strict mode means any conflict is an error
        with pytest.raises(ValueError):
            github_archive_repo_events.recursive_dict_merge(target, source)

    def test_modifies_target_in_place(self):
        """Test that the function modifies target dict in place."""
        target = {"a": 1}
        source = {"b": 2}
        target_id = id(target)

        github_archive_repo_events.recursive_dict_merge(target, source)

        # Should be the same object
        assert id(target) == target_id
        assert target == {"a": 1, "b": 2}
