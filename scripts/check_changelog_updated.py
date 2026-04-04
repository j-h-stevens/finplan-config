"""CI enforcement: fail if config YAML files changed without a new changes/ entry.

Run from repo root::

    python scripts/check_changelog_updated.py

Exits 0 when the changelog is up-to-date or when no config files were modified.
Exits 1 (and prints guidance) when config files changed but no new entry exists.

In CI this is called after checkout so it compares the working tree against
the merge base of the PR (origin/main).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# Only allow safe git ref characters — no spaces, semicolons, or shell metacharacters.
_SAFE_REF_RE = re.compile(r"^[\w/.\-]+$")

# Config directories that require a changelog entry when modified
CONFIG_DIRS = ("finplan_config/tax_years", "finplan_config/capital_market")
CONFIG_SUFFIXES = (".yaml", ".yml")

CHANGES_DIR = REPO_ROOT / "changes"


def _git_changed_files(base_ref: str = "origin/main") -> list[str]:
    """Return files changed relative to base_ref, or all tracked changes if unavailable."""
    if not _SAFE_REF_RE.match(base_ref):
        print(f"ERROR: base_ref {base_ref!r} contains invalid characters (expected {_SAFE_REF_RE.pattern})")
        sys.exit(1)
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref, "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        )
        return result.stdout.strip().splitlines()
    except subprocess.CalledProcessError:
        # Fallback: staged + unstaged changes (useful in local runs without a remote)
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        return result.stdout.strip().splitlines()


def _is_config_file(path: str) -> bool:
    p = Path(path)
    return (
        any(str(p).startswith(d) for d in CONFIG_DIRS)
        and p.suffix in CONFIG_SUFFIXES
    )


def _new_changes_entries(base_ref: str = "origin/main") -> list[str]:
    """Return new files added under changes/ since base_ref."""
    changed = _git_changed_files(base_ref)
    return [f for f in changed if f.startswith("changes/") and f.endswith(".yaml")]


def main() -> int:
    base_ref = "origin/main"

    changed = _git_changed_files(base_ref)
    config_changes = [f for f in changed if _is_config_file(f)]
    new_entries = _new_changes_entries(base_ref)

    if not config_changes:
        print("check_changelog: no config files modified — OK")
        return 0

    print(f"check_changelog: {len(config_changes)} config file(s) modified:")
    for f in config_changes:
        print(f"  {f}")

    if not new_entries:
        print(
            "\nERROR: Config files were modified but no new entry was added under changes/.\n"
            "Add a file like:\n"
            "  changes/YYYY-MM-DD-short-description.yaml\n"
            "with fields: version, date, type, affected_configs, summary,\n"
            "             downstream_action_required, notes\n"
            "Then re-run: python scripts/aggregate_changelog.py"
        )
        return 1

    print(f"\ncheck_changelog: {len(new_entries)} new changes/ entry(ies) found — OK")
    for e in new_entries:
        print(f"  {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
