"""Functional data-correctness tests for bundled finplan-config YAML files.

These tests verify that the shipped configuration data is internally consistent:
bracket contiguity, monotonic RMD divisors, allocation weight sums, correlation
matrix symmetry, and required key presence.  They complement the security tests
(test_security.py) which focus on the loading layer rather than data content.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from finplan_config import ConfigRegistry, _MIN_VALID_YEAR, _MAX_VALID_YEAR


@pytest.fixture(autouse=True)
def reset_registry():
    ConfigRegistry.reset()
    yield
    ConfigRegistry.reset()


@pytest.fixture
def registry():
    return ConfigRegistry.get()


# ──────────────────────────────────────────────────────────────────────────────
# IRS Limits
# ──────────────────────────────────────────────────────────────────────────────


class TestIRSLimits:
    def test_required_top_level_keys_present(self, registry):
        limits = registry.irs_limits(year=2024)
        required = {"traditional_401k", "ira", "social_security", "hsa"}
        missing = required - limits.keys()
        assert not missing, f"IRS limits missing keys: {missing}"

    def test_traditional_401k_limit_positive(self, registry):
        limits = registry.irs_limits(year=2024)
        assert limits["traditional_401k"]["limit"] > 0

    def test_catchup_limit_less_than_total(self, registry):
        limits = registry.irs_limits(year=2024)
        entry = limits["traditional_401k"]
        assert entry["catch_up"] < entry["limit_over_50"]

    def test_ira_catchup_adds_up(self, registry):
        limits = registry.irs_limits(year=2024)
        ira = limits["ira"]
        assert math.isclose(ira["limit"] + ira["catch_up"], ira["limit_over_50"])

    def test_hsa_family_greater_than_self(self, registry):
        limits = registry.irs_limits(year=2024)
        hsa = limits["hsa"]
        assert hsa["limit_family"] > hsa["limit_self"]


# ──────────────────────────────────────────────────────────────────────────────
# Federal Brackets
# ──────────────────────────────────────────────────────────────────────────────

FILING_STATUSES = {
    "ordinary": {
        "single",
        "married_filing_jointly",
        "married_filing_separately",
        "head_of_household",
    },
    "ltcg": {
        "single",
        "married_filing_jointly",
        "married_filing_separately",
        "head_of_household",
    },
}


class TestFederalBrackets:
    def test_all_filing_statuses_present(self, registry):
        brackets = registry.federal_brackets_np(year=2024)
        for category, statuses in FILING_STATUSES.items():
            missing = statuses - brackets[category].keys()
            assert not missing, f"federal_brackets[{category}] missing: {missing}"

    def test_brackets_are_contiguous(self, registry):
        brackets = registry.federal_brackets_np(year=2024)
        for category, by_status in brackets.items():
            for status, arr in by_status.items():
                name = f"federal_brackets.{category}.{status}"
                for i in range(1, len(arr)):
                    prev_max = arr[i - 1, 1]
                    curr_min = arr[i, 0]
                    if not math.isinf(prev_max):
                        assert abs(prev_max - curr_min) < 0.01, (
                            f"{name} bracket {i}: gap between {prev_max} and {curr_min}"
                        )

    def test_last_bracket_extends_to_inf(self, registry):
        brackets = registry.federal_brackets_np(year=2024)
        for category, by_status in brackets.items():
            for status, arr in by_status.items():
                assert math.isinf(arr[-1, 1]), (
                    f"federal_brackets.{category}.{status}: last bracket max is not inf"
                )

    def test_rates_in_valid_range(self, registry):
        brackets = registry.federal_brackets_np(year=2024)
        for category, by_status in brackets.items():
            for status, arr in by_status.items():
                rates = arr[:, 2]
                assert np.all(rates >= 0) and np.all(rates <= 1), (
                    f"federal_brackets.{category}.{status} has rate outside [0, 1]"
                )

    def test_brackets_start_at_zero(self, registry):
        brackets = registry.federal_brackets_np(year=2024)
        for category, by_status in brackets.items():
            for status, arr in by_status.items():
                assert arr[0, 0] == 0.0, (
                    f"federal_brackets.{category}.{status}: first bracket min != 0"
                )

    def test_ltcg_has_fewer_or_equal_brackets_than_ordinary(self, registry):
        brackets = registry.federal_brackets_np(year=2024)
        for status in brackets["ltcg"]:
            ltcg_n = len(brackets["ltcg"][status])
            ordinary_n = len(brackets["ordinary"][status])
            assert ltcg_n <= ordinary_n, (
                f"LTCG brackets ({ltcg_n}) > ordinary brackets ({ordinary_n}) for {status}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# State Tax
# ──────────────────────────────────────────────────────────────────────────────

# All 50 states + DC
_ALL_STATE_CODES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "DC",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
}


class TestStateTax:
    def test_all_states_and_dc_present(self, registry):
        data = registry.state_tax_data(year=2024)
        missing = _ALL_STATE_CODES - data.keys()
        assert not missing, f"state_tax missing states: {sorted(missing)}"

    def test_entry_count_is_51(self, registry):
        data = registry.state_tax_data(year=2024)
        assert len(data) == 51

    def test_no_income_tax_states_are_none_type(self, registry):
        no_tax_states = {"FL", "NV", "SD", "TN", "TX", "WA", "WY"}
        data = registry.state_tax_data(year=2024)
        for state in no_tax_states:
            if state in data:
                tax_type = data[state][0]
                assert tax_type == "none", (
                    f"{state} is a no-income-tax state but has type={tax_type!r}"
                )

    def test_flat_rate_entries_have_positive_rate(self, registry):
        data = registry.state_tax_data(year=2024)
        for code, entry in data.items():
            if entry[0] == "flat":
                assert entry[1] > 0, f"State {code}: flat rate is not positive"
                assert entry[1] < 1, f"State {code}: flat rate >= 1.0"

    def test_graduated_entries_have_numpy_array(self, registry):
        data = registry.state_tax_data(year=2024)
        for code, entry in data.items():
            if entry[0] == "graduated":
                assert isinstance(entry[1], np.ndarray), (
                    f"State {code}: graduated brackets should be ndarray"
                )
                assert entry[1].ndim == 2 and entry[1].shape[1] == 3, (
                    f"State {code}: brackets array should be (N, 3)"
                )


# ──────────────────────────────────────────────────────────────────────────────
# RMD Table
# ──────────────────────────────────────────────────────────────────────────────


class TestRMDTable:
    def test_age_coverage_72_to_120(self, registry):
        table = registry.rmd_table(year=2024)
        missing = [age for age in range(72, 121) if age not in table]
        assert not missing, f"RMD table missing ages: {missing}"

    def test_divisors_monotonically_decreasing(self, registry):
        table = registry.rmd_table(year=2024)
        ages = sorted(table.keys())
        for i in range(1, len(ages)):
            assert table[ages[i]] < table[ages[i - 1]], (
                f"RMD divisor at age {ages[i]} ({table[ages[i]]}) is not less than "
                f"age {ages[i - 1]} ({table[ages[i - 1]]})"
            )

    def test_divisors_are_positive(self, registry):
        table = registry.rmd_table(year=2024)
        for age, divisor in table.items():
            assert divisor > 0, f"RMD divisor at age {age} is not positive"

    def test_age_72_divisor_greater_than_age_120(self, registry):
        table = registry.rmd_table(year=2024)
        assert table[72] > table[120]


# ──────────────────────────────────────────────────────────────────────────────
# Capital Market Assumptions
# ──────────────────────────────────────────────────────────────────────────────


class TestCapitalMarket:
    @pytest.mark.parametrize("horizon", ["10yr", "20yr"])
    def test_allocation_weights_sum_to_one(self, registry, horizon):
        cm = registry.capital_market(horizon=horizon)
        if "allocations" not in cm:
            pytest.skip(f"No allocations in {horizon}")
        for name, weights in cm["allocations"].items():
            total = sum(weights)
            assert abs(total - 1.0) < 1e-4, (
                f"capital_market({horizon}) allocation '{name}' sums to {total}, expected 1.0"
            )

    @pytest.mark.parametrize("horizon", ["10yr", "20yr"])
    def test_correlation_matrix_is_symmetric(self, registry, horizon):
        cm = registry.capital_market(horizon=horizon)
        if "_correlation_matrix" not in cm:
            pytest.skip(f"No correlation matrix in {horizon}")
        mat = cm["_correlation_matrix"]
        assert np.allclose(mat, mat.T, atol=1e-6), (
            f"capital_market({horizon}) correlation matrix is not symmetric"
        )

    @pytest.mark.parametrize("horizon", ["10yr", "20yr"])
    def test_correlation_diagonal_is_ones(self, registry, horizon):
        cm = registry.capital_market(horizon=horizon)
        if "_correlation_matrix" not in cm:
            pytest.skip(f"No correlation matrix in {horizon}")
        mat = cm["_correlation_matrix"]
        assert np.allclose(np.diag(mat), 1.0, atol=1e-6), (
            f"capital_market({horizon}) correlation matrix diagonal is not all 1.0"
        )

    @pytest.mark.parametrize("horizon", ["10yr", "20yr"])
    def test_correlation_values_in_range(self, registry, horizon):
        cm = registry.capital_market(horizon=horizon)
        if "_correlation_matrix" not in cm:
            pytest.skip(f"No correlation matrix in {horizon}")
        mat = cm["_correlation_matrix"]
        assert np.all(mat >= -1.0) and np.all(mat <= 1.0), (
            f"capital_market({horizon}) correlation values outside [-1, 1]"
        )

    @pytest.mark.parametrize("horizon", ["10yr", "20yr"])
    def test_asset_class_arrays_consistent_length(self, registry, horizon):
        cm = registry.capital_market(horizon=horizon)
        n = len(cm["_names"])
        assert len(cm["_arithmetic_return"]) == n
        assert len(cm["_geometric_return"]) == n
        assert len(cm["_volatility"]) == n

    @pytest.mark.parametrize("horizon", ["10yr", "20yr"])
    def test_volatilities_are_positive(self, registry, horizon):
        cm = registry.capital_market(horizon=horizon)
        assert np.all(cm["_volatility"] > 0), (
            f"capital_market({horizon}) has non-positive volatility"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Plan Defaults
# ──────────────────────────────────────────────────────────────────────────────


class TestPlanDefaults:
    def test_required_sections_present(self, registry):
        defaults = registry.plan_defaults()
        required = {
            "plan_assumptions",
            "monte_carlo",
            "scenario_defaults",
            "conversion_layer",
        }
        missing = required - defaults.keys()
        assert not missing, f"plan_defaults missing sections: {missing}"

    def test_inflation_rate_in_plausible_range(self, registry):
        rate = registry.plan_defaults()["plan_assumptions"]["inflation_rate"]
        assert 0.0 < rate < 0.20, f"inflation_rate {rate} is implausible"

    def test_monte_carlo_iterations_positive(self, registry):
        iters = registry.plan_defaults()["monte_carlo"]["default_trials"]
        assert iters > 0

    def test_life_expectancy_in_range(self, registry):
        le = registry.plan_defaults()["plan_assumptions"]["life_expectancy_client"]
        assert 70 <= le <= 120, f"life_expectancy_client {le} is implausible"

    def test_confidence_level_in_range(self, registry):
        cl = registry.plan_defaults()["plan_assumptions"]["confidence_level"]
        assert 0.0 < cl <= 1.0, f"confidence_level {cl} outside (0, 1]"


# ──────────────────────────────────────────────────────────────────────────────
# Year validation
# ──────────────────────────────────────────────────────────────────────────────


class TestYearValidation:
    def test_year_below_min_raises(self, registry):
        with pytest.raises(ValueError, match="supported range"):
            registry.irs_limits(year=_MIN_VALID_YEAR - 1)

    def test_year_above_max_raises(self, registry):
        with pytest.raises(ValueError, match="supported range"):
            registry.irs_limits(year=_MAX_VALID_YEAR + 1)

    def test_year_at_min_boundary_does_not_raise(self, registry):
        # Should not raise ValueError (may raise FileNotFoundError for missing year)
        try:
            registry.irs_limits(year=_MIN_VALID_YEAR)
        except ValueError as e:
            pytest.fail(f"Should not raise ValueError for year={_MIN_VALID_YEAR}: {e}")
        except FileNotFoundError:
            pass  # Expected — no data for that year, but validation passed

    def test_year_at_max_boundary_does_not_raise(self, registry):
        try:
            registry.irs_limits(year=_MAX_VALID_YEAR)
        except ValueError as e:
            pytest.fail(f"Should not raise ValueError for year={_MAX_VALID_YEAR}: {e}")
        except FileNotFoundError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Changelog
# ──────────────────────────────────────────────────────────────────────────────


class TestChangelog:
    def test_changes_log_yaml_parseable(self):
        from finplan_config.changelog import _load_log

        entries = _load_log()
        assert isinstance(entries, list)
        assert len(entries) > 0

    def test_get_changes_since_older_version_returns_entries(self):
        from finplan_config.changelog import get_changes_since

        entries = get_changes_since("2023.0.0")
        assert len(entries) > 0

    def test_get_changes_since_latest_returns_empty(self):
        from finplan_config.changelog import get_changes_since, _load_log

        entries = _load_log()
        latest = max(e["version"] for e in entries)
        result = get_changes_since(latest)
        assert result == []

    def test_has_breaking_changes_since_returns_false_for_current(self):
        from finplan_config.changelog import has_breaking_changes_since

        assert not has_breaking_changes_since("2023.0.0")


# ──────────────────────────────────────────────────────────────────────────────
# Thread safety
# ──────────────────────────────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_get_creates_single_instance(self):
        """50 threads calling ConfigRegistry.get() concurrently must all get
        the same singleton instance — no duplicate initialisation."""
        ConfigRegistry.reset()
        instances = []
        errors = []

        def worker():
            try:
                instances.append(ConfigRegistry.get())
            except Exception as e:
                errors.append(e)

        import threading

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Threads raised errors: {errors}"
        assert len(instances) == 50
        # All threads must have received the identical object
        assert all(inst is instances[0] for inst in instances), (
            "ConfigRegistry.get() returned different instances across threads"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Plan defaults validation
# ──────────────────────────────────────────────────────────────────────────────


class TestPlanDefaultsValidation:
    def test_validate_all_includes_plan_defaults(self):
        """validate_all() must pass for the bundled plan_defaults.yaml."""
        from pathlib import Path
        from finplan_config.validators import validate_all

        config_dir = Path(__file__).parent.parent / "finplan_config"
        # Should not raise
        validate_all(config_dir)

    def test_validate_plan_defaults_rejects_missing_section(self):
        from finplan_config.validators import validate_plan_defaults, ConfigError

        bad = {
            "plan_assumptions": {"inflation_rate": 0.03},
            "monte_carlo": {"default_trials": 1000},
        }
        with pytest.raises(ConfigError, match="missing required sections"):
            validate_plan_defaults(bad)

    def test_validate_plan_defaults_rejects_implausible_inflation(self):
        from finplan_config.validators import validate_plan_defaults, ConfigError

        bad = {
            "plan_assumptions": {"inflation_rate": 5.0},  # 500% — implausible
            "monte_carlo": {"default_trials": 1000},
            "scenario_defaults": {},
            "conversion_layer": {},
        }
        with pytest.raises(ConfigError, match="inflation_rate"):
            validate_plan_defaults(bad)

    def test_validate_plan_defaults_rejects_zero_trials(self):
        from finplan_config.validators import validate_plan_defaults, ConfigError

        bad = {
            "plan_assumptions": {"inflation_rate": 0.03},
            "monte_carlo": {"default_trials": 0},
            "scenario_defaults": {},
            "conversion_layer": {},
        }
        with pytest.raises(ConfigError, match="default_trials"):
            validate_plan_defaults(bad)


# ──────────────────────────────────────────────────────────────────────────────
# Config manifest (audit trail)
# ──────────────────────────────────────────────────────────────────────────────


class TestConfigManifest:
    def test_manifest_has_required_keys(self, registry):
        manifest = registry.config_manifest(year=2024)
        for key in ("tax_year", "config_hash", "package_version", "generated_at"):
            assert key in manifest, f"config_manifest missing key: {key}"

    def test_manifest_hash_is_deterministic(self, registry):
        m1 = registry.config_manifest(year=2024)
        m2 = registry.config_manifest(year=2024)
        assert m1["config_hash"] == m2["config_hash"]

    def test_manifest_hash_is_16_hex_chars(self, registry):
        h = registry.config_manifest(year=2024)["config_hash"]
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_manifest_tax_year_matches_request(self, registry):
        manifest = registry.config_manifest(year=2024)
        assert manifest["tax_year"] == 2024

    def test_manifest_generated_at_is_utc_iso(self, registry):
        ts = registry.config_manifest(year=2024)["generated_at"]
        assert ts.endswith("Z")
        # Must parse without error
        import datetime

        datetime.datetime.fromisoformat(ts.rstrip("Z"))


# ──────────────────────────────────────────────────────────────────────────────
# Staleness warning
# ──────────────────────────────────────────────────────────────────────────────


class TestStalenessWarning:
    def test_stale_warning_fires_when_year_is_behind(self):
        """If current_year > latest_bundled_year, a UserWarning must be emitted."""
        import datetime
        import warnings
        from finplan_config import _warn_if_stale
        from pathlib import Path

        config_dir = Path(__file__).parent.parent / "finplan_config"
        current_year = datetime.date.today().year
        latest_bundled = 2024  # only 2024 data is present

        if current_year > latest_bundled:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _warn_if_stale(config_dir)
            assert any(
                issubclass(w.category, UserWarning)
                and "stale" in str(w.message).lower()
                for w in caught
            ), "Expected a UserWarning about stale config but none was raised"
        else:
            # No warning expected when running in 2024
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _warn_if_stale(config_dir)
            stale_warnings = [w for w in caught if "stale" in str(w.message).lower()]
            assert not stale_warnings, "Unexpected stale warning for current year"

    def test_no_warning_when_config_dir_has_no_tax_years(self, tmp_path):
        """Missing tax_years dir must not crash — just returns silently."""
        import warnings
        from finplan_config import _warn_if_stale

        (tmp_path / "tax_years").mkdir()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_if_stale(tmp_path)
        assert not caught


# ──────────────────────────────────────────────────────────────────────────────
# available_tax_years()
# ──────────────────────────────────────────────────────────────────────────────


class TestAvailableTaxYears:
    def test_returns_list_of_ints(self, registry):
        years = registry.available_tax_years()
        assert isinstance(years, list)
        assert all(isinstance(y, int) for y in years)

    def test_2024_is_present(self, registry):
        assert 2024 in registry.available_tax_years()

    def test_years_are_sorted(self, registry):
        years = registry.available_tax_years()
        assert years == sorted(years)

    def test_empty_dir_returns_empty_list(self, tmp_path):
        from finplan_config import ConfigRegistry

        (tmp_path / "tax_years").mkdir()
        (tmp_path / "capital_market").mkdir()
        cr = ConfigRegistry.__new__(ConfigRegistry)
        cr._config_dir = tmp_path
        cr._default_year = 2024
        assert cr.available_tax_years() == []
