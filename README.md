# connors-sr

> Part of the [Connors Trading System](https://github.com/marcelohack/connors-playground)

## Overview

Support & Resistance calculator with multiple technical analysis methods for identifying key price levels. Provides a programmatic API and integrates with the playground CLI for interactive analysis.

## Features

- **4 Built-in Methods**: Pivot Points, Fractal Analysis, VWAP Zones, Volume Profile
- **External Method Support**: Load custom SR calculation methods via decorator pattern
- **Data Integration**: Works seamlessly with connors-datafetch for market data
- **Visualization**: Built-in Plotly charts for price levels
- **File Storage**: Save/load calculations in JSON and HTML formats

## Installation

```bash
pip install connors-sr
```

### Local Development

**Prerequisites**: Python 3.13, [pyenv](https://github.com/pyenv/pyenv) + [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)

```bash
# 1. Create and activate a virtual environment
pyenv virtualenv 3.13 connors-sr
pyenv activate connors-sr

# 2. Install connors packages from local checkouts (not on PyPI)
pip install -e ../datafetch

# 3. Install with dev dependencies
pip install -e ".[dev]"
```

A `.python-version` file is included so pyenv auto-activates when you `cd` into this directory.

## Quick Start

```python
from connors_sr.services.sr_service import SRService, SRCalculationRequest
from connors_sr.core.support_resistance import SRMethod

# Initialize service
service = SRService()

# Create calculation request
request = SRCalculationRequest(
    ticker="AAPL",
    method=SRMethod.PIVOT_POINTS,
    datasource="yfinance",
    start="2024-01-01",
    end="2024-12-31"
)

# Calculate SR levels
result = service.calculate_sr(request)

# Access results
print(f"Found {len(result.sr_result.levels)} support/resistance levels")
for level in result.sr_result.levels:
    print(f"{level.level_type}: ${level.level:.2f} (strength: {level.strength:.2f})")
```

## CLI Usage

The S/R calculator CLI is part of [connors-playground](https://github.com/marcelohack/connors-playground):

```bash
# Pivot Points calculation
python -m connors.cli.sr_calculator --ticker AAPL --method pivot_points --timespan 6M

# Fractal analysis with custom parameters
python -m connors.cli.sr_calculator --ticker MSFT --method fractal \
  --method-params "lookback:7;min_strength:0.4"

# VWAP Zones with plotting
python -m connors.cli.sr_calculator --ticker NVDA --method vwap_zones --timespan YTD --plot

# Volume Profile with results saving
python -m connors.cli.sr_calculator --ticker TSLA --method volume_profile \
  --save-results --save-plot

# Show available parameters
python -m connors.cli.sr_calculator --method pivot_points --show-method-params

# External method
python -m connors.cli.sr_calculator --ticker AAPL \
  --external-method ~/.connors/sr_methods/test_sr_method.py \
  --method-params "calculation_method:ema_bands;ema_period:30" --timespan 3M

# Different markets and data sources
python -m connors.cli.sr_calculator --ticker BHP --method pivot_points \
  --market australia --datasource yfinance --timespan 1Y

# Using dataset file
python -m connors.cli.sr_calculator --ticker CUSTOM --method vwap_zones \
  --dataset-file my_data.csv --plot

# List methods and saved results
python -m connors.cli.sr_calculator --list-methods
python -m connors.cli.sr_calculator --list-saved
```

## Available Methods

| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `pivot_points` | Classic pivot point calculations | `period`, `include_midpoints` |
| `fractal` | Fractal-based support/resistance | `lookback`, `min_strength` |
| `vwap_zones` | VWAP-based price zones | `period`, `std_devs`, `min_volume_ratio` |
| `volume_profile` | Volume profile price levels | `price_bins`, `min_volume_pct`, `lookback_periods` |

## Custom SR Calculators

Create custom calculators using the decorator pattern:

```python
from connors_sr.core.registry import registry
from connors_sr.core.support_resistance import BaseSRCalculator, SRResult

@registry.register_sr_method("my_custom_method")
class MyCustomCalculator(BaseSRCalculator):
    def calculate(self, data, **params):
        # Your calculation logic
        return SRResult(...)

    def get_default_parameters(self):
        return {"param1": 10, "param2": 2.0}

    def get_parameter_info(self):
        return {
            "param1": {"type": int, "description": "..."},
            "param2": {"type": float, "description": "..."}
        }
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=connors_sr
```

## Related Packages

| Package | Description | Links |
|---------|-------------|-------|
| [connors-playground](https://github.com/marcelohack/connors-playground) | CLI + Streamlit UI (integration hub) | [README](https://github.com/marcelohack/connors-playground#readme) |
| [connors-core](https://github.com/marcelohack/connors-core) | Registry, config, indicators, metrics | [README](https://github.com/marcelohack/connors-core#readme) |
| [connors-backtest](https://github.com/marcelohack/connors-backtest) | Backtesting service + built-in strategies | [README](https://github.com/marcelohack/connors-backtest#readme) |
| [connors-strategies](https://github.com/marcelohack/connors-strategies) | Trading strategy collection (private) | — |
| [connors-screener](https://github.com/marcelohack/connors-screener) | Stock screening system | [README](https://github.com/marcelohack/connors-screener#readme) |
| [connors-datafetch](https://github.com/marcelohack/connors-datafetch) | Multi-source data downloader | [README](https://github.com/marcelohack/connors-datafetch#readme) |
| [connors-regime](https://github.com/marcelohack/connors-regime) | Market regime detection | [README](https://github.com/marcelohack/connors-regime#readme) |
| [connors-bots](https://github.com/marcelohack/connors-bots) | Automated trading bots | [README](https://github.com/marcelohack/connors-bots#readme) |

## License

MIT
