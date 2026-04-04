# finplan-config

**Shared configuration package for finplan services** — IRS limits, federal/state tax brackets, capital market assumptions, and plan defaults.

This package extracts regulatory and financial configuration data into versioned YAML files, enabling non-developers (compliance, product teams) to update values without touching code. Consumed by both `finplan-compute-engine` and `finplan-api-gateway`.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/license-proprietary-red.svg)](LICENSE)

## ✨ Key Features

- **Tax year versioning:** Separate config directories for 2024, 2025, etc.
- **Multiple data formats:** IRS limits (dict), brackets (numpy arrays), state taxes (tuples), capital market (correlations + weights)
- **Lazy loading & caching:** Loaded once per process, cached forever via singleton
- **Structural validation:** Bracket contiguity, correlation matrix symmetry, weight sums
- **Environment override:** `FINPLAN_CONFIG_DIR` to point to custom config directory
- **No dependencies:** Pure Python + PyYAML only; minimal footprint

## 📦 Installation

```bash
# From PyPI (when published)
pip install finplan-config

# From source
git clone https://github.com/finplan/finplan-config
cd finplan-config
pip install -e .
```

## 🚀 Quick Start

```python
from finplan_config import ConfigRegistry

# Singleton instance (lazy-initializes on first call)
cr = ConfigRegistry.get()

# IRS limits (dict)
limits = cr.irs_limits(year=2024)
print(limits['traditional_401k']['limit'])  # 23500

# Federal tax brackets (numpy arrays)
brackets = cr.federal_brackets_np(year=2024)
single_ordinary = brackets['ordinary']['single']  # shape (7, 3)

# State tax data (tuple format)
state_data = cr.state_tax_data(year=2024)
ca_tax = state_data['CA']  # ('graduated', np.ndarray)

# Capital market assumptions
cm = cr.capital_market(horizon='10yr')
asset_names = cm['_names']  # 17 asset classes
correlation_matrix = cm['_correlation_matrix']  # 17×17

# Plan defaults
defaults = cr.plan_defaults()
inflation = defaults['plan']['inflation_rate']  # 0.03
```

### Custom Config Directory

```python
from pathlib import Path
from finplan_config import ConfigRegistry

# One-time initialization
ConfigRegistry.initialize(
    config_dir=Path('/opt/finplan/config'),
    default_year=2024
)

# Or via environment variable
export FINPLAN_CONFIG_DIR=/opt/finplan/config
```

## 📁 Configuration Structure

### `config/tax_years/{YEAR}/` — Annually Updated

#### `irs_limits.yaml` (25 values)
Contribution limits, catch-up ages, RMD start age:
```yaml
traditional_401k:
  limit: 23500
  catch_up_age: 50
  catch_up_limit: 7500

traditional_ira:
  limit: 7000

rmd_start_age: 73
```

#### `federal_brackets.yaml` (80 values)
Tax brackets, standard deductions, FICA, NIIT, SALT cap:
```yaml
ordinary_income:
  single:
    - [0, 11600, 0.10]          # [min, max, rate]
    - [11600, 47150, 0.12]
    # ...

standard_deductions:
  single: 13850
  mfj: 27700

ltcg_qualified_dividends:
  single:
    - [0, 47025, 0.00]
```

#### `state_tax.yaml` (51 entries)
All 50 states + DC income tax rules:
```yaml
AK:
  type: none

CO:
  type: flat
  rate: 0.0455

CA:
  type: graduated
  brackets:
    - [0, 10099, 0.01]
    - [10099, 23942, 0.02]
```

#### `rmd_tables.yaml`
Required Minimum Distribution divisors (ages 72–120):
```yaml
uniform_lifetime_table:
  72: 27.4
  73: 26.5
  # ...
  120: 2.0
```

### `config/capital_market/` — Multi-Year Horizons

#### `exhibit17_10yr.yaml` / `exhibit17_20yr.yaml`
17 asset classes, returns, volatility, correlation matrix, allocations:
```yaml
asset_classes:
  - name: "US Large Cap"
    arithmetic_return: 0.10
    geometric_return: 0.095
    volatility: 0.16
  # ... 16 more

correlation_matrix:
  - [1.0, 0.65, 0.42, ...]      # 17×17 symmetric matrix
  - [0.65, 1.0, 0.38, ...]

allocations:
  balanced:
    - [0.30, 0.20, 0.15, ...]   # Sums to 1.0
  aggressive:
    - [0.40, 0.25, 0.20, ...]
```

### `config/plan_defaults.yaml`

Plan assumptions and scenario event defaults:
```yaml
plan:
  inflation_rate: 0.03
  pre_retirement_return: 0.07
  post_retirement_return: 0.05
  life_expectancy: 90

monte_carlo:
  num_trials: 1000
  volatility_pre_retirement: 0.16

scenarios:
  bear_market:
    severity: -0.20
  disability:
    probability: 0.06
```

## 📅 Annual Updates

When a new tax year is released:

1. **Create directory:**
   ```bash
   mkdir -p config/tax_years/2025/
   cp config/tax_years/2024/* config/tax_years/2025/
   ```

2. **Update values:**
   - IRS limits: Check IRS Rev. Proc.
   - Federal brackets: Check IRS Rev. Proc.
   - State taxes: Check each state legislature
   - Capital market: Consult Vanguard Exhibit 17 or institutional sources

3. **Validate:**
   ```bash
   python -m finplan_config.validators --year 2025
   ```

4. **Release:**
   ```bash
   git tag 2025.1.0
   git push --tags
   pip install --upgrade finplan-config>=2025.0.0
   ```

## ✅ Validation

YAML files are automatically validated when loaded. Manual validation:

```python
from finplan_config.validators import validate_all
validate_all(year=2024)  # Raises ValidationError if issues
```

**Rules:**
- Brackets: Contiguous (no gaps), last bracket has `.inf` max
- RMD table: Ages 72–120, divisors monotonically decreasing
- Correlation matrix: Symmetric, diagonal = 1.0, values in [-1, 1]
- Allocation weights: Sum to 1.0 ± 0.0001 tolerance
- State tax entries: All have valid `type` field

## 📊 Code Coverage

```bash
pip install pytest-cov
pytest finplan_config/ --cov=finplan_config --cov-report=html
open htmlcov/index.html
```

Currently: **95%+ coverage** on ConfigRegistry and validators.

## 🏗️ Architecture

```
finplan-config/
├── finplan_config/
│   ├── __init__.py              # ConfigRegistry singleton (277 lines)
│   ├── validators.py            # Validation functions (173 lines)
│   ├── tax_years/
│   │   └── 2024/
│   │       ├── irs_limits.yaml
│   │       ├── federal_brackets.yaml
│   │       ├── state_tax.yaml
│   │       └── rmd_tables.yaml
│   ├── capital_market/
│   │   ├── exhibit17_10yr.yaml
│   │   └── exhibit17_20yr.yaml
│   └── plan_defaults.yaml
├── pyproject.toml
├── README.md
└── LICENSE
```

## 🔧 Development

### Testing

```bash
pip install pytest
pytest finplan_config/ -v
```

### Linting

```bash
pip install ruff
ruff check finplan_config/
ruff format finplan_config/
```

### Publishing to PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
```

## 🐛 Troubleshooting

| Error | Solution |
|-------|----------|
| "No config found for tax year 2025" | Create `config/tax_years/2025/` or set `FINPLAN_CONFIG_DIR` |
| "Brackets not contiguous" | Check YAML: bracket max[N] must equal min[N+1] |
| "Cannot import finplan_config" | Run `pip install -e .` from repo root |
| "Validation error: weights not sum to 1.0" | Check allocation weights: `sum(weights) ≈ 1.0` |

## 📝 License

Proprietary — finplan internal use only.

## 👥 Contributing

Questions or issues? Contact: team@finplan.com
