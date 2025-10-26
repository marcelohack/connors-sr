# Connors SR (Support & Resistance Calculator)

A standalone Python package for identifying Support & Resistance levels using various technical analysis methods.

## Features

- **Multiple Calculation Methods**:
  - Pivot Points (Classic, Fibonacci, Camarilla)
  - Fractal-based levels
  - VWAP Zones
  - Volume Profile

- **External Method Support**: Load custom SR calculation methods via decorator pattern
- **Data Integration**: Works seamlessly with connors-datafetch for market data
- **Visualization**: Built-in plotly charts for price levels
- **File Storage**: Save/load calculations in JSON and HTML formats

## Installation

```bash
pip install connors-sr
```

For development:
```bash
git clone https://github.com/marcelohack/connors-sr.git
cd connors-sr
pip install -e ".[dev]"
```

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

## Available Methods

- `pivot_points`: Classic pivot point calculations
- `fractal`: Fractal-based support/resistance
- `vwap_zones`: VWAP-based zones
- `volume_profile`: Volume profile levels

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

## Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- plotly >= 5.17.0
- connors-datafetch >= 0.1.0

## License

MIT License - see LICENSE file for details
