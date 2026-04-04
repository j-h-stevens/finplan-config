"""Configuration registry for finplan-py.

Loads YAML config files once at first access, converts bracket data to numpy
arrays, and caches everything for the lifetime of the process.

Override the config directory with the ``FINPLAN_CONFIG_DIR`` environment
variable.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import ClassVar

import numpy as np
import yaml

from .validators import validate_all

logger = logging.getLogger(__name__)

_INF = float("inf")

# Expected subdirectories that must exist for a valid config directory.
_REQUIRED_SUBDIRS = ("tax_years", "capital_market")

# Year range guard — catches obviously wrong inputs (e.g. year=1800, year=9999)
# before they silently fall back to the default year.
_MIN_VALID_YEAR = 1990
_MAX_VALID_YEAR = 2100


def _validated_config_dir(path: Path) -> Path:
    """Resolve and validate a config directory path from FINPLAN_CONFIG_DIR.

    Guards against path traversal: the resolved path must be an existing
    directory that contains the expected finplan config structure.

    Raises ValueError for any suspicious or invalid path.
    """
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"FINPLAN_CONFIG_DIR={path!r} does not exist or cannot be resolved: {exc}"
        ) from exc

    if not resolved.is_dir():
        raise ValueError(f"FINPLAN_CONFIG_DIR={resolved!r} is not a directory")

    # Verify the directory looks like a finplan config tree (not an arbitrary path).
    missing = [sub for sub in _REQUIRED_SUBDIRS if not (resolved / sub).is_dir()]
    if missing:
        raise ValueError(
            f"FINPLAN_CONFIG_DIR={resolved!r} is missing expected subdirectories: "
            f"{missing}. Is this a valid finplan config directory?"
        )

    return resolved


# Sentinel so we don't confuse "not loaded" with "loaded but empty"
_NOT_LOADED = object()


def _yaml_load(path: Path) -> dict:
    """Load a single YAML file and return a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def _bracket_list_to_np(rows: list) -> np.ndarray:
    """Convert a list of ``[min, max, rate]`` rows to an ``(N, 3)`` float64 array.

    Handles YAML ``.inf`` which PyYAML deserialises as Python ``float('inf')``.
    """
    return np.array(rows, dtype=np.float64)


def _state_tax_tuple(entry: dict) -> tuple:
    """Convert a YAML state-tax dict to the tuple format the engine expects.

    Returns:
        ``("none",)`` | ``("flat", rate)`` | ``("graduated", np.ndarray)``
    """
    tax_type = entry["type"]
    if tax_type == "none":
        return ("none",)
    elif tax_type == "flat":
        return ("flat", float(entry["rate"]))
    else:
        brackets = _bracket_list_to_np(entry["brackets"])
        return ("graduated", brackets)


class ConfigRegistry:
    """Singleton that loads and caches YAML configuration.

    Thread-safe: a class-level lock guards singleton creation via double-checked
    locking. After ``initialize()`` completes, all access is read-only on
    immutable data (dicts + numpy arrays).
    """

    _instance: ClassVar[ConfigRegistry | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _config_dir: Path
    _default_year: int

    # Caches
    _irs_limits_cache: dict[int, dict]
    _federal_brackets_cache: dict[int, dict]
    _federal_brackets_np_cache: dict[int, dict]
    _state_tax_cache: dict[int, dict]
    _rmd_cache: dict[int, dict]
    _capital_market_cache: dict[str, dict]
    _plan_defaults_cache: dict | None

    def __init__(self, config_dir: Path, default_year: int = 2024) -> None:
        self._config_dir = config_dir
        self._default_year = default_year
        self._irs_limits_cache = {}
        self._federal_brackets_cache = {}
        self._federal_brackets_np_cache = {}
        self._state_tax_cache = {}
        self._rmd_cache = {}
        self._capital_market_cache = {}
        self._plan_defaults_cache = None

    # ------------------------------------------------------------------
    # Singleton lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def initialize(
        cls,
        config_dir: Path | None = None,
        default_year: int = 2024,
        *,
        validate: bool = True,
    ) -> "ConfigRegistry":
        """Create (or replace) the singleton with the given config directory."""
        if config_dir is None:
            env_dir = os.environ.get("FINPLAN_CONFIG_DIR")
            if env_dir:
                config_dir = _validated_config_dir(Path(env_dir))
            else:
                config_dir = Path(__file__).parent
        instance = cls(config_dir, default_year)
        if validate:
            validate_all(config_dir, default_year)
        # Assign without acquiring _lock: callers that need thread safety (get())
        # already hold the lock before calling initialize().  Explicit direct calls
        # to initialize() are expected to run before concurrent access begins.
        cls._instance = instance
        return instance

    @classmethod
    def get(cls) -> "ConfigRegistry":
        """Return the singleton, lazy-initializing if necessary.

        Uses double-checked locking so concurrent callers never create more
        than one instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls.initialize()
        return cls._instance  # type: ignore[return-value]

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Year resolution
    # ------------------------------------------------------------------

    def _resolve_year(self, year: int | None) -> int:
        """Return the requested year, or the default year.

        Raises ``ValueError`` for years outside the supported range
        ``[_MIN_VALID_YEAR, _MAX_VALID_YEAR]`` to surface obviously wrong inputs
        early rather than silently falling back to the default year.
        """
        if year is None:
            return self._default_year
        if not (_MIN_VALID_YEAR <= year <= _MAX_VALID_YEAR):
            raise ValueError(
                f"Tax year {year} is outside the supported range "
                f"[{_MIN_VALID_YEAR}, {_MAX_VALID_YEAR}]"
            )
        return year

    def _year_dir(self, year: int) -> Path:
        """Return the directory for a tax year, falling back to nearest available."""
        target = self._config_dir / "tax_years" / str(year)
        if target.is_dir():
            return target
        # Fall back to default year
        fallback = self._config_dir / "tax_years" / str(self._default_year)
        if fallback.is_dir():
            logger.warning(
                "Tax year %d not found; falling back to %d", year, self._default_year
            )
            return fallback
        raise FileNotFoundError(
            f"No config found for tax year {year} or default {self._default_year}"
        )

    # ------------------------------------------------------------------
    # IRS limits
    # ------------------------------------------------------------------

    def irs_limits(self, year: int | None = None) -> dict:
        """Return IRS limits dict for the given tax year."""
        yr = self._resolve_year(year)
        if yr not in self._irs_limits_cache:
            path = self._year_dir(yr) / "irs_limits.yaml"
            self._irs_limits_cache[yr] = _yaml_load(path)
        return self._irs_limits_cache[yr]

    # ------------------------------------------------------------------
    # Federal brackets (raw dict)
    # ------------------------------------------------------------------

    def federal_brackets(self, year: int | None = None) -> dict:
        """Return the raw federal brackets dict for the given tax year."""
        yr = self._resolve_year(year)
        if yr not in self._federal_brackets_cache:
            path = self._year_dir(yr) / "federal_brackets.yaml"
            self._federal_brackets_cache[yr] = _yaml_load(path)
        return self._federal_brackets_cache[yr]

    # ------------------------------------------------------------------
    # Federal brackets (numpy arrays)
    # ------------------------------------------------------------------

    def federal_brackets_np(self, year: int | None = None) -> dict:
        """Return federal brackets as numpy arrays keyed by category.

        Returns::

            {
                "ordinary": {"single": np.ndarray, ...},
                "ltcg": {"single": np.ndarray, ...},
            }
        """
        yr = self._resolve_year(year)
        if yr not in self._federal_brackets_np_cache:
            raw = self.federal_brackets(yr)
            ordinary = {}
            for status, rows in raw["ordinary_income"].items():
                ordinary[status] = _bracket_list_to_np(rows)
            ltcg = {}
            for status, rows in raw["ltcg_qualified_dividends"].items():
                ltcg[status] = _bracket_list_to_np(rows)
            self._federal_brackets_np_cache[yr] = {
                "ordinary": ordinary,
                "ltcg": ltcg,
            }
        return self._federal_brackets_np_cache[yr]

    # ------------------------------------------------------------------
    # State tax data
    # ------------------------------------------------------------------

    def state_tax_data(self, year: int | None = None) -> dict[str, tuple]:
        """Return state tax data in the tuple format the engine expects."""
        yr = self._resolve_year(year)
        if yr not in self._state_tax_cache:
            path = self._year_dir(yr) / "state_tax.yaml"
            raw = _yaml_load(path)
            result = {}
            for code, entry in raw.items():
                result[code] = _state_tax_tuple(entry)
            self._state_tax_cache[yr] = result
        return self._state_tax_cache[yr]

    # ------------------------------------------------------------------
    # RMD tables
    # ------------------------------------------------------------------

    def rmd_table(self, year: int | None = None) -> dict[int, float]:
        """Return RMD Uniform Lifetime Table as ``{age: divisor}``."""
        yr = self._resolve_year(year)
        if yr not in self._rmd_cache:
            path = self._year_dir(yr) / "rmd_tables.yaml"
            raw = _yaml_load(path)
            # YAML keys might be strings; ensure int keys
            self._rmd_cache[yr] = {
                int(k): float(v) for k, v in raw["uniform_lifetime_table"].items()
            }
        return self._rmd_cache[yr]

    # ------------------------------------------------------------------
    # Capital market assumptions
    # ------------------------------------------------------------------

    def capital_market(self, horizon: str = "10yr") -> dict:
        """Return capital market assumptions for a given horizon."""
        if horizon not in self._capital_market_cache:
            path = self._config_dir / "capital_market" / f"exhibit17_{horizon}.yaml"
            raw = _yaml_load(path)
            # Convert lists to numpy arrays for downstream use
            asset_classes = raw["asset_classes"]
            raw["_names"] = [a["name"] for a in asset_classes]
            raw["_arithmetic_return"] = np.array(
                [a["arithmetic_return"] for a in asset_classes]
            )
            raw["_geometric_return"] = np.array(
                [a["geometric_return"] for a in asset_classes]
            )
            raw["_volatility"] = np.array([a["volatility"] for a in asset_classes])
            if "correlation_matrix" in raw:
                raw["_correlation_matrix"] = np.array(
                    raw["correlation_matrix"], dtype=np.float64
                )
            if "allocations" in raw:
                for name, weights in raw["allocations"].items():
                    raw[f"_weights_{name}"] = np.array(weights, dtype=np.float64)
            self._capital_market_cache[horizon] = raw
        return self._capital_market_cache[horizon]

    # ------------------------------------------------------------------
    # Plan defaults
    # ------------------------------------------------------------------

    def plan_defaults(self) -> dict:
        """Return plan and scenario defaults."""
        if self._plan_defaults_cache is None:
            path = self._config_dir / "plan_defaults.yaml"
            self._plan_defaults_cache = _yaml_load(path)
        return self._plan_defaults_cache
