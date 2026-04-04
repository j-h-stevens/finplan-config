# finplan-config

Shared configuration package for finplan services.

## Installation

```bash
pip install -e finplan-config/
```

## Usage

```python
from finplan_config import ConfigRegistry

cr = ConfigRegistry.get()
limits = cr.irs_limits(year=2024)
brackets = cr.federal_brackets_np(year=2024)
```

