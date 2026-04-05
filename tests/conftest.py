"""
Pytest configuration for finplan-config tests.

Suppresses the expected stale-config UserWarning early in pytest startup —
before any fixtures run. This is necessary because some CI environments set
PYTHONWARNINGS=error, which would otherwise convert the warning to an exception
in every test that calls ConfigRegistry.get().

The stale-config warning is intentional and correct: we only ship 2024 tax data,
so any year > 2024 will always trigger it. Tests should not fail because of it.
"""

import warnings


def pytest_configure(config: object) -> None:
    """Register warning filters before any test collection or fixture setup."""
    warnings.filterwarnings(
        "ignore",
        message=r"finplan-config.*stale config",
        category=UserWarning,
    )
