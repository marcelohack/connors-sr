# CLAUDE.md

This file provides guidance to Claude Code when working with the connors-sr package.

## Package Overview

**connors-sr** is a standalone Support & Resistance calculator package that identifies key price levels using various technical analysis methods.

- **Package name**: `connors-sr` (PyPI/GitHub)
- **Module name**: `connors_sr` (Python imports)
- **Repository**: https://github.com/marcelohack/connors-sr

## Architecture

### Core Components

- **`connors_sr/core/support_resistance.py`**: Core SR calculation logic
  - `SRMethod` enum: Available calculation methods
  - `SRLevel`, `SRResult`: Data containers
  - `BaseSRCalculator`: Base class for calculators
  - Built-in calculators: `PivotPointsCalculator`, `FractalCalculator`, `VolumeProfileCalculator`, `VWAPZonesCalculator`

- **`connors_sr/services/sr_service.py`**: High-level service interface
  - `SRService`: Main service class
  - `SRCalculationRequest`, `SRServiceResult`: Request/response models
  - Data fetching integration via connors-datafetch
  - File storage and visualization

- **`connors_sr/core/registry.py`**: External method registration
  - Decorator pattern for custom SR calculators
  - `@registry.register_sr_method("name")`

### Key Features

1. **Multiple Calculation Methods**:
   - Pivot Points (classic, fibonacci, camarilla)
   - Fractal-based levels
   - VWAP zones
   - Volume profile

2. **External Method Support**:
   - Users can create custom SR calculators
   - Register via `@registry.register_sr_method()` decorator
   - Load from external Python files

3. **Data Integration**:
   - Uses `connors-datafetch` for market data
   - Supports multiple data sources (yfinance, polygon, etc.)
   - Can load from saved datasets

4. **Storage & Visualization**:
   - Saves results to `{appHome}/sr_calculations/`
   - JSON format for results
   - HTML plotly charts for visualization

## Dependencies

- **Required**:
  - pandas >= 2.0.0
  - numpy >= 1.24.0
  - plotly >= 5.17.0
  - connors-datafetch >= 0.1.0

- **Development**:
  - pytest, pytest-cov, pytest-mock
  - black, isort, mypy

## Testing

Run tests with:
```bash
pytest tests/ -v
```

## Code Standards

- Python 3.13+
- Type hints where practical
- Black formatting (line length 88)
- isort for import sorting
- Comprehensive docstrings
