"""Cross-repo change log utilities for finplan-config.

Downstream repos (finplan-compute-engine, finplan-api-gateway) use this module
to detect breaking changes between their pinned version and the installed version.

Typical usage at service startup::

    from finplan_config.changelog import has_breaking_changes_since

    PINNED_VERSION = "2024.1.0"
    if has_breaking_changes_since(PINNED_VERSION):
        import sys
        print("WARNING: finplan-config has breaking changes since", PINNED_VERSION)
        print("Run: python -c 'from finplan_config.changelog import print_changes_since;"
              f" print_changes_since(\"{PINNED_VERSION}\")'")

The change log lives in ``changes_log.yaml`` at the package root and is also
shipped inside the installed package so offline use works without git.
"""

from __future__ import annotations

import logging
from functools import total_ordering
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CHANGES_LOG_PATH = Path(__file__).parent.parent / "changes_log.yaml"

# Entry types ordered by severity (used for filtering)
_SEVERITY_ORDER = ["feat", "fix", "data", "security", "breaking"]


_MAX_VERSION_LEN = 50


def _pad_parts(
    a: tuple[int, ...], b: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Pad the shorter tuple with trailing zeros so comparisons are length-neutral.

    Without this, ``(2024, 1) < (2024, 1, 0)`` would return ``True`` in Python
    because shorter tuples sort before longer ones when all common elements are
    equal — breaking CalVer equivalence like ``"2024.1" == "2024.1.0"``.
    """
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)), b + (0,) * (n - len(b))


@total_ordering
class _Version:
    """Minimal semantic-ish version comparator for CalVer (YYYY.N.0) strings.

    ``"2024.1"`` and ``"2024.1.0"`` compare equal; ``"2024.1.1" > "2024.1.0"``.
    """

    def __init__(self, version_str: str) -> None:
        if len(version_str) > _MAX_VERSION_LEN:
            raise ValueError(
                f"Version string is too long ({len(version_str)} chars, max {_MAX_VERSION_LEN}): "
                f"{version_str[:20]!r}..."
            )
        self._str = version_str
        try:
            self._parts = tuple(int(x) for x in version_str.split("."))
        except ValueError:
            raise ValueError(f"Cannot parse version string: {version_str!r}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Version):
            return NotImplemented
        a, b = _pad_parts(self._parts, other._parts)
        return a == b

    def __lt__(self, other: _Version) -> bool:
        a, b = _pad_parts(self._parts, other._parts)
        return a < b

    def __repr__(self) -> str:
        return f"_Version({self._str!r})"


def _load_log() -> list[dict]:
    """Load and return the list of entries from changes_log.yaml.

    Returns an empty list (never raises) if the file is missing, unreadable,
    or contains invalid YAML — so downstream service startup checks degrade
    gracefully rather than crashing the process.
    """
    if not _CHANGES_LOG_PATH.exists():
        logger.warning("changes_log.yaml not found at %s", _CHANGES_LOG_PATH)
        return []
    try:
        with open(_CHANGES_LOG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("changes_log.yaml could not be parsed: %s", exc)
        return []
    if not isinstance(data, dict) or "entries" not in data:
        logger.warning("changes_log.yaml has unexpected structure")
        return []
    return data.get("entries") or []


def get_changes_since(version: str) -> list[dict]:
    """Return all change entries with a version strictly greater than ``version``.

    Args:
        version: The pinned version string your repo last verified against
                 (e.g. ``"2024.1.0"``).

    Returns:
        List of change entry dicts, sorted oldest-first.  Each entry has:
        ``version``, ``date``, ``type``, ``affected_configs``, ``summary``,
        ``downstream_action_required``, ``notes``.
    """
    threshold = _Version(version)
    entries = _load_log()
    return [e for e in entries if _Version(str(e["version"])) > threshold]


def has_breaking_changes_since(version: str) -> bool:
    """Return ``True`` if any entry since ``version`` has ``type == 'breaking'``.

    Args:
        version: The pinned version string your repo last verified against.

    Returns:
        ``True`` if a breaking change exists in any newer entry; ``False`` otherwise.
    """
    return any(e.get("type") == "breaking" for e in get_changes_since(version))


def has_downstream_actions_since(version: str) -> bool:
    """Return ``True`` if any entry since ``version`` requires downstream action."""
    return any(e.get("downstream_action_required") for e in get_changes_since(version))


def print_changes_since(version: str) -> None:
    """Print a human-readable summary of changes since ``version`` to stdout."""
    entries = get_changes_since(version)
    if not entries:
        print(f"No changes since {version}.")
        return
    print(f"Changes in finplan-config since {version}:")
    print("-" * 60)
    for e in entries:
        action = " [ACTION REQUIRED]" if e.get("downstream_action_required") else ""
        print(f"  {e['date']}  v{e['version']}  [{e['type'].upper()}]{action}")
        summary = str(e.get("summary", "")).strip().replace("\n", " ")
        print(f"    {summary}")
        if e.get("notes"):
            notes = str(e["notes"]).strip().replace("\n", " ")
            print(f"    Note: {notes}")
    print("-" * 60)
