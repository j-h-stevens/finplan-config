"""Lightweight structural validation for YAML configuration files.

No Pydantic dependency — pure Python checks that run at config-load time
and in CI to catch transcription errors before deployment.
"""

from __future__ import annotations

import math
from pathlib import Path


class ConfigError(Exception):
    """Raised when a configuration file fails validation."""


def validate_brackets(rows: list, name: str) -> None:
    """Validate a list of ``[min, max, rate]`` bracket rows."""
    if not rows:
        raise ConfigError(f"{name}: empty bracket list")
    for i, row in enumerate(rows):
        if len(row) != 3:
            raise ConfigError(f"{name} bracket {i}: expected 3 values, got {len(row)}")
        lo, hi, rate = row
        if lo < 0:
            raise ConfigError(f"{name} bracket {i}: min {lo} is negative")
        if hi <= lo and not math.isinf(hi):
            raise ConfigError(f"{name} bracket {i}: max {hi} <= min {lo}")
        if rate < 0 or rate > 1:
            raise ConfigError(f"{name} bracket {i}: rate {rate} out of [0, 1]")
    # Check contiguity
    for i in range(1, len(rows)):
        prev_max = rows[i - 1][1]
        curr_min = rows[i][0]
        if not math.isinf(prev_max) and abs(prev_max - curr_min) > 0.01:
            raise ConfigError(
                f"{name} bracket {i}: min {curr_min} != prior max {prev_max}"
            )
    # Last bracket must extend to inf
    last_max = rows[-1][1]
    if not math.isinf(last_max):
        raise ConfigError(f"{name}: last bracket max is {last_max}, expected inf")


def validate_state_tax(data: dict) -> None:
    """Validate state tax data (all 51 entries)."""
    required_types = {"none", "flat", "graduated"}
    for code, entry in data.items():
        tax_type = entry.get("type")
        if tax_type not in required_types:
            raise ConfigError(f"State {code}: unknown type {tax_type!r}")
        if tax_type == "flat" and "rate" not in entry:
            raise ConfigError(f"State {code}: flat type missing 'rate'")
        if tax_type == "graduated":
            if "brackets" not in entry:
                raise ConfigError(f"State {code}: graduated type missing 'brackets'")
            # State brackets use [min, max, rate] format
            for i, row in enumerate(entry["brackets"]):
                if len(row) != 3:
                    raise ConfigError(
                        f"State {code} bracket {i}: expected 3 values, got {len(row)}"
                    )


def validate_rmd_table(table: dict) -> None:
    """Validate RMD Uniform Lifetime Table."""
    if not table:
        raise ConfigError("RMD table is empty")
    # Check ages 72-120 present
    for age in range(72, 121):
        if age not in table:
            raise ConfigError(f"RMD table: missing age {age}")
    # Check divisors monotonically decreasing
    ages = sorted(table.keys())
    for i in range(1, len(ages)):
        if table[ages[i]] >= table[ages[i - 1]]:
            raise ConfigError(
                f"RMD table: divisor at age {ages[i]} ({table[ages[i]]}) "
                f">= divisor at age {ages[i - 1]} ({table[ages[i - 1]]})"
            )


def validate_correlation_matrix(matrix: list) -> None:
    """Validate a correlation matrix (list of lists)."""
    n = len(matrix)
    if n == 0:
        raise ConfigError("Correlation matrix is empty")
    for i, row in enumerate(matrix):
        if len(row) != n:
            raise ConfigError(
                f"Correlation matrix row {i}: expected {n} values, got {len(row)}"
            )
        if abs(row[i] - 1.0) > 1e-6:
            raise ConfigError(
                f"Correlation matrix: diagonal[{i}] = {row[i]}, expected 1.0"
            )
        for j, val in enumerate(row):
            if val < -1.0 or val > 1.0:
                raise ConfigError(
                    f"Correlation matrix[{i}][{j}] = {val}, out of [-1, 1]"
                )
    # Check symmetry
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-6:
                raise ConfigError(
                    f"Correlation matrix: [{i}][{j}]={matrix[i][j]} != "
                    f"[{j}][{i}]={matrix[j][i]}"
                )


def validate_allocation_weights(weights: list, name: str) -> None:
    """Validate allocation weights sum to 1.0."""
    total = sum(weights)
    if abs(total - 1.0) > 1e-4:
        raise ConfigError(f"Allocation {name}: weights sum to {total}, expected 1.0")


def validate_all(config_dir: Path, default_year: int = 2024) -> None:
    """Run all validations against the config directory.

    Called automatically by ``ConfigRegistry.initialize()`` and can be
    invoked from CI.
    """
    import yaml

    year_dir = config_dir / "tax_years" / str(default_year)
    if not year_dir.is_dir():
        raise ConfigError(f"Tax year directory not found: {year_dir}")

    # -- IRS limits: just check it loads --
    irs_path = year_dir / "irs_limits.yaml"
    if irs_path.exists():
        with open(irs_path) as f:
            yaml.safe_load(f)

    # -- Federal brackets --
    fb_path = year_dir / "federal_brackets.yaml"
    if fb_path.exists():
        with open(fb_path) as f:
            fb = yaml.safe_load(f)
        for status, rows in fb.get("ordinary_income", {}).items():
            validate_brackets(rows, f"federal_brackets.ordinary_income.{status}")
        for status, rows in fb.get("ltcg_qualified_dividends", {}).items():
            validate_brackets(rows, f"federal_brackets.ltcg.{status}")

    # -- State tax --
    st_path = year_dir / "state_tax.yaml"
    if st_path.exists():
        with open(st_path) as f:
            st = yaml.safe_load(f)
        validate_state_tax(st)

    # -- RMD tables --
    rmd_path = year_dir / "rmd_tables.yaml"
    if rmd_path.exists():
        with open(rmd_path) as f:
            rmd = yaml.safe_load(f)
        table = {int(k): float(v) for k, v in rmd["uniform_lifetime_table"].items()}
        validate_rmd_table(table)

    # -- Capital market --
    cm_dir = config_dir / "capital_market"
    for horizon in ("10yr", "20yr"):
        cm_path = cm_dir / f"exhibit17_{horizon}.yaml"
        if cm_path.exists():
            with open(cm_path) as f:
                cm = yaml.safe_load(f)
            if "correlation_matrix" in cm:
                validate_correlation_matrix(cm["correlation_matrix"])
            if "allocations" in cm:
                for name, weights in cm["allocations"].items():
                    validate_allocation_weights(weights, name)
