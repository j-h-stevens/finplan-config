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
    """Validate state tax data (all 51 entries).

    Keys starting with ``_`` (e.g. ``_metadata``) are silently skipped so
    that audit-trail blocks can coexist with state entries in the same YAML.
    """
    required_types = {"none", "flat", "graduated"}
    for code, entry in data.items():
        if code.startswith("_"):
            continue  # skip internal metadata keys
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


def validate_plan_defaults(data: dict) -> None:
    """Validate plan_defaults.yaml structure and key assumption ranges."""
    required_sections = {"plan_assumptions", "monte_carlo", "scenario_defaults", "conversion_layer"}
    missing = required_sections - data.keys()
    if missing:
        raise ConfigError(f"plan_defaults.yaml missing required sections: {sorted(missing)}")

    assumptions = data["plan_assumptions"]
    rate = assumptions.get("inflation_rate")
    if rate is None:
        raise ConfigError("plan_defaults.yaml plan_assumptions.inflation_rate is missing")
    if not (0 < rate < 0.5):
        raise ConfigError(
            f"plan_defaults.yaml plan_assumptions.inflation_rate={rate} is implausible "
            f"(expected 0 < rate < 0.5)"
        )

    mc = data["monte_carlo"]
    trials = mc.get("default_trials")
    if trials is None or trials <= 0:
        raise ConfigError(
            f"plan_defaults.yaml monte_carlo.default_trials={trials!r} must be a positive integer"
        )


def validate_metadata(data: dict, filename: str) -> None:
    """Validate the ``_metadata`` block present in each config YAML.

    The ``_metadata`` block records the IRS source, publication date, and
    last-verified date, providing an audit trail that bundled values match
    published IRS data.

    This validation is intentionally lenient (only warns on missing blocks) to
    avoid breaking configs that pre-date the metadata requirement.
    """
    import logging
    import re

    logger = logging.getLogger(__name__)

    meta = data.get("_metadata")
    if meta is None:
        logger.warning(
            "%s is missing a _metadata block. Add source, date_published, and "
            "last_verified to maintain an IRS audit trail.",
            filename,
        )
        return

    if not isinstance(meta, dict):
        raise ConfigError(f"{filename} _metadata must be a mapping, got {type(meta).__name__}")

    # Validate date fields are ISO-format strings (YYYY-MM-DD)
    iso_date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for field in ("date_published", "last_verified"):
        value = meta.get(field)
        if value and not iso_date_re.match(str(value)):
            raise ConfigError(
                f"{filename} _metadata.{field}={value!r} is not a valid ISO date (YYYY-MM-DD)"
            )


def validate_all(config_dir: Path, default_year: int = 2024) -> None:
    """Run all validations against the config directory.

    Called automatically by ``ConfigRegistry.initialize()`` and can be
    invoked from CI.
    """
    import yaml

    year_dir = config_dir / "tax_years" / str(default_year)
    if not year_dir.is_dir():
        raise ConfigError(f"Tax year directory not found: {year_dir}")

    # -- IRS limits: load + metadata --
    irs_path = year_dir / "irs_limits.yaml"
    if irs_path.exists():
        with open(irs_path, encoding="utf-8") as f:
            irs_data = yaml.safe_load(f)
        validate_metadata(irs_data, "irs_limits.yaml")

    # -- Federal brackets --
    fb_path = year_dir / "federal_brackets.yaml"
    if fb_path.exists():
        with open(fb_path, encoding="utf-8") as f:
            fb = yaml.safe_load(f)
        validate_metadata(fb, "federal_brackets.yaml")
        for status, rows in fb.get("ordinary_income", {}).items():
            validate_brackets(rows, f"federal_brackets.ordinary_income.{status}")
        for status, rows in fb.get("ltcg_qualified_dividends", {}).items():
            validate_brackets(rows, f"federal_brackets.ltcg.{status}")

    # -- State tax --
    st_path = year_dir / "state_tax.yaml"
    if st_path.exists():
        with open(st_path, encoding="utf-8") as f:
            st = yaml.safe_load(f)
        validate_metadata(st, "state_tax.yaml")
        validate_state_tax(st)

    # -- RMD tables --
    rmd_path = year_dir / "rmd_tables.yaml"
    if rmd_path.exists():
        with open(rmd_path, encoding="utf-8") as f:
            rmd = yaml.safe_load(f)
        validate_metadata(rmd, "rmd_tables.yaml")
        table = {int(k): float(v) for k, v in rmd["uniform_lifetime_table"].items()}
        validate_rmd_table(table)

    # -- Capital market --
    cm_dir = config_dir / "capital_market"
    for horizon in ("10yr", "20yr"):
        cm_path = cm_dir / f"exhibit17_{horizon}.yaml"
        if cm_path.exists():
            with open(cm_path, encoding="utf-8") as f:
                cm = yaml.safe_load(f)
            if "correlation_matrix" in cm:
                validate_correlation_matrix(cm["correlation_matrix"])
            if "allocations" in cm:
                for name, weights in cm["allocations"].items():
                    validate_allocation_weights(weights, name)

    # -- Plan defaults --
    pd_path = config_dir / "plan_defaults.yaml"
    if pd_path.exists():
        with open(pd_path, encoding="utf-8") as f:
            pd_data = yaml.safe_load(f)
        validate_plan_defaults(pd_data)
