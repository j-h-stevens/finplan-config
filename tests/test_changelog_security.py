"""Security and correctness tests for the changelog module.

Covers:
- Version comparator correctness for CalVer strings of differing component counts
- Length guard on version strings
- Graceful handling of missing or corrupted changes_log.yaml
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from finplan_config.changelog import (
    _Version,
    _load_log,
    get_changes_since,
    has_breaking_changes_since,
)


# ──────────────────────────────────────────────────────────────────────────────
# Version comparator correctness
# ──────────────────────────────────────────────────────────────────────────────


class TestVersionComparator:
    def test_calver_with_and_without_patch_are_equal(self):
        """'2024.1' and '2024.1.0' must compare equal (CalVer equivalence)."""
        assert _Version("2024.1") == _Version("2024.1.0")

    def test_calver_not_fooled_by_length(self):
        """'2024.1.1' must be strictly greater than '2024.1.0', not equal."""
        assert _Version("2024.1.1") > _Version("2024.1.0")
        assert not (_Version("2024.1.1") == _Version("2024.1.0"))

    def test_shorter_tuple_not_less_when_logically_equal(self):
        """Before the fix, (2024, 1) < (2024, 1, 0) was True — that was wrong."""
        assert not (_Version("2024.1") < _Version("2024.1.0"))

    def test_major_version_ordering(self):
        assert _Version("2025.1.0") > _Version("2024.1.0")

    def test_minor_version_ordering(self):
        assert _Version("2024.2.0") > _Version("2024.1.0")

    def test_equal_versions(self):
        assert _Version("2024.1.0") == _Version("2024.1.0")

    def test_three_part_less_than(self):
        assert _Version("2024.1.0") < _Version("2024.1.1")

    def test_version_string_too_long_raises(self):
        """Version strings longer than 50 characters must be rejected."""
        long_version = "2024." + "1" * 60
        with pytest.raises(ValueError, match="too long"):
            _Version(long_version)

    def test_version_string_exactly_at_limit_accepted(self):
        """A 50-character version string is within the allowed limit."""
        # "2024." + 45 chars = 50 total
        ok_version = "2024." + "1" * 45
        v = _Version(ok_version)
        assert v._str == ok_version

    def test_non_numeric_parts_raise(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _Version("2024.x.0")

    def test_get_changes_since_uses_padded_comparison(self):
        """get_changes_since('2024.1') should not return the '2024.1.0' entry."""
        # '2024.1' and '2024.1.0' are equal, so no entries should be returned
        # for an already-matching version.
        entries = get_changes_since("2024.1")
        # All entries at or before 2024.1.0 should be excluded
        for e in entries:
            assert _Version(str(e["version"])) > _Version("2024.1"), (
                f"Entry {e['version']} should not appear in changes since '2024.1'"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Robustness: missing or corrupted changes_log.yaml
# ──────────────────────────────────────────────────────────────────────────────


class TestChangeLogRobustness:
    def test_missing_log_returns_empty_list(self):
        """If changes_log.yaml doesn't exist, _load_log() returns [] — no exception."""
        with patch(
            "finplan_config.changelog._CHANGES_LOG_PATH", Path("/nonexistent/path.yaml")
        ):
            result = _load_log()
        assert result == []

    def test_corrupted_log_returns_empty_list(self, tmp_path):
        """Truncated / invalid YAML in changes_log.yaml returns [] — no exception."""
        bad_log = tmp_path / "changes_log.yaml"
        bad_log.write_text(": this: is: not: valid: yaml: {{{", encoding="utf-8")
        with patch("finplan_config.changelog._CHANGES_LOG_PATH", bad_log):
            result = _load_log()
        assert result == []

    def test_log_with_wrong_structure_returns_empty_list(self, tmp_path):
        """A YAML file that parses but lacks 'entries' returns []."""
        bad_log = tmp_path / "changes_log.yaml"
        bad_log.write_text("schema_version: 1\n# no entries key\n", encoding="utf-8")
        with patch("finplan_config.changelog._CHANGES_LOG_PATH", bad_log):
            result = _load_log()
        assert result == []

    def test_get_changes_since_with_missing_log_returns_empty(self):
        """get_changes_since() on a missing log is safe — returns []."""
        with patch(
            "finplan_config.changelog._CHANGES_LOG_PATH", Path("/nonexistent/path.yaml")
        ):
            result = get_changes_since("2024.1.0")
        assert result == []

    def test_has_breaking_changes_with_missing_log_returns_false(self):
        """has_breaking_changes_since() on a missing log returns False safely."""
        with patch(
            "finplan_config.changelog._CHANGES_LOG_PATH", Path("/nonexistent/path.yaml")
        ):
            assert not has_breaking_changes_since("2024.1.0")
