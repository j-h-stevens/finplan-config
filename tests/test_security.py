"""Security tests for finplan-config.

Tests guard against:
- Path traversal via FINPLAN_CONFIG_DIR
- YAML injection (safe_load only)
- Malformed config doesn't crash with sensitive output
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from finplan_config import ConfigRegistry
from finplan_config import _validated_config_dir


@pytest.fixture(autouse=True)
def reset_registry():
    ConfigRegistry.reset()
    yield
    ConfigRegistry.reset()


class TestPathTraversalPrevention:
    """FINPLAN_CONFIG_DIR must not allow arbitrary path traversal."""

    def test_rejects_nonexistent_path(self, monkeypatch):
        monkeypatch.setenv("FINPLAN_CONFIG_DIR", "/nonexistent/path/to/nowhere")
        with pytest.raises(ValueError, match="does not exist"):
            ConfigRegistry.initialize()

    def test_rejects_file_path(self, tmp_path, monkeypatch):
        f = tmp_path / "not_a_dir.txt"
        f.write_text("data")
        monkeypatch.setenv("FINPLAN_CONFIG_DIR", str(f))
        with pytest.raises(ValueError, match="is not a directory"):
            ConfigRegistry.initialize()

    def test_rejects_dir_without_required_structure(self, tmp_path, monkeypatch):
        """A real directory that doesn't have tax_years/ and capital_market/ is rejected."""
        monkeypatch.setenv("FINPLAN_CONFIG_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="missing expected subdirectories"):
            ConfigRegistry.initialize()

    def test_rejects_path_traversal_attempt(self, tmp_path, monkeypatch):
        """Path with .. that resolves outside expected dirs is rejected."""
        evil = tmp_path / ".." / ".." / "etc"
        monkeypatch.setenv("FINPLAN_CONFIG_DIR", str(evil))
        with pytest.raises(ValueError):
            ConfigRegistry.initialize()

    def test_accepts_valid_config_dir(self):
        """The bundled config directory passes validation."""
        package_dir = Path(__file__).parent.parent / "finplan_config"
        result = _validated_config_dir(package_dir)
        assert result.is_dir()
        assert (result / "tax_years").is_dir()
        assert (result / "capital_market").is_dir()

    def test_resolved_path_returned(self):
        """Symlinks are resolved before validation."""
        package_dir = Path(__file__).parent.parent / "finplan_config"
        with tempfile.TemporaryDirectory() as td:
            link = Path(td) / "config_link"
            link.symlink_to(package_dir)
            result = _validated_config_dir(link)
            assert result == package_dir.resolve()


class TestYAMLSafety:
    """YAML must always use safe_load — never allow arbitrary Python objects."""

    def test_yaml_load_uses_safe_load(self, tmp_path):
        """A YAML file with a Python object tag must be rejected, not executed."""
        # Create a config dir with the right structure
        (tmp_path / "tax_years" / "2024").mkdir(parents=True)
        (tmp_path / "capital_market").mkdir()

        malicious_yaml = tmp_path / "tax_years" / "2024" / "irs_limits.yaml"
        # yaml.load() with Loader=Loader would execute this; safe_load raises
        malicious_yaml.write_text("!!python/object/apply:os.system ['echo pwned']")

        from finplan_config import _yaml_load
        import yaml

        with pytest.raises(yaml.constructor.ConstructorError):
            _yaml_load(malicious_yaml)

    def test_normal_yaml_loads_correctly(self):
        """Normal YAML data loads without error."""
        from finplan_config import _yaml_load
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\nnumber: 42\n")
            name = f.name

        try:
            data = _yaml_load(Path(name))
            assert data == {"key": "value", "number": 42}
        finally:
            os.unlink(name)


class TestConfigRegistryIsolation:
    """Singleton must be properly reset between test runs."""

    def test_reset_clears_instance(self):
        ConfigRegistry.get()  # creates instance
        assert ConfigRegistry._instance is not None
        ConfigRegistry.reset()
        assert ConfigRegistry._instance is None

    def test_lazy_init_uses_bundled_config(self):
        """Default init (no env var) uses the bundled YAML files."""
        cr = ConfigRegistry.get()
        limits = cr.irs_limits(year=2024)
        assert "traditional_401k" in limits
        assert limits["traditional_401k"]["limit"] > 0

    def test_year_fallback_does_not_expose_arbitrary_paths(self, monkeypatch):
        """Year resolution falls back to default year — not to arbitrary filesystem."""
        cr = ConfigRegistry.get()
        # Requesting a far-future year should fall back to 2024, not error with
        # a path that could be influenced by input
        data = cr.irs_limits(year=9999)
        assert "traditional_401k" in data
