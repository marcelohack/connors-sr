# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

**connors-sr** is a standalone Support & Resistance calculator package that identifies key price levels using various technical analysis methods.

- **Package name**: `connors-sr` (PyPI/GitHub)
- **Module name**: `connors_sr` (Python imports)
- **Repository**: https://github.com/marcelohack/connors-sr

## Development Commands

### Environment Setup
```bash
# Install package in development mode with dev dependencies
pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests with verbose output
pytest tests/ -v

# Run tests with coverage report
pytest tests/ -v --cov=connors_sr --cov-report=term-missing

# Run a single test file
pytest tests/test_sr_calculator.py -v

# Run a specific test
pytest tests/test_sr_calculator.py::TestSRCalculators::test_pivot_points_calculator -v
```

### Code Quality
```bash
# Format code with black
black connors_sr tests

# Sort imports with isort
isort connors_sr tests

# Type checking with mypy
mypy connors_sr

# Run all quality checks together
black connors_sr tests && isort connors_sr tests && mypy connors_sr
```

### Building and Distribution
```bash
# Build the package
python -m build

# Install locally for testing
pip install -e .
```

## Architecture

### Core Components

The package follows a layered architecture with clear separation between calculation logic, service orchestration, and extensibility:

**`connors_sr/core/support_resistance.py`** - Core SR calculation logic (1000+ lines)
- `SRMethod` enum: Defines available built-in methods (PIVOT_POINTS, FRACTAL, VWAP_ZONES, VOLUME_PROFILE)
- Data models: `SRLevel` (individual level), `SRResult` (calculation results)
- `BaseSRCalculator`: Abstract base class with common functionality (`_validate_data`, `_add_sr_columns`)
- Built-in calculators implementing `BaseSRCalculator`:
  - `PivotPointsCalculator`: Traditional pivot points with swing highs/lows
  - `FractalCalculator`: Fractal-based S/R using lookback windows
  - `VWAPZonesCalculator`: VWAP with standard deviation bands
  - `VolumeProfileCalculator`: Price-level volume distribution analysis

**`connors_sr/services/sr_service.py`** - High-level service orchestration (885 lines)
- `SRService`: Main service coordinating data fetch, calculation, storage, and visualization
- Request/response models: `SRCalculationRequest`, `SRServiceResult`
- Data integration: Uses `connors-datafetch` service for market data across multiple sources
- File operations: Saves results to `{CONNORS_HOME}/sr_calculations/{method}/{ticker}_{market}_{start}_{end}.json`
- Visualization: Generates interactive Plotly charts with candlesticks, volume, and S/R levels
- External method loading: `load_external_method(file_path)` dynamically imports custom calculators

**`connors_sr/core/registry.py`** - Extensibility mechanism
- `SRMethodRegistry`: Manages external SR calculator registration
- Decorator pattern: `@registry.register_sr_method("name")` for custom calculators
- Global `registry` instance for application-wide access

### Data Flow Pattern

1. **Request Creation**: User creates `SRCalculationRequest` with ticker, method, parameters
2. **Data Acquisition**: `SRService._download_data()` or `_load_dataset_file()` fetches OHLCV data
3. **Column Normalization**: `_prepare_dataframe_for_sr_calculation()` ensures title-case columns (Open, High, Low, Close, Volume)
4. **Calculation**: Selected calculator's `calculate()` method processes data
5. **Result Processing**: `_save_results()` and `_generate_plot()` persist outputs
6. **Return**: `SRServiceResult` returned with results, file paths, success status

### Extension Pattern

Custom SR calculators must:
- Inherit from `BaseSRCalculator` (recommended) or implement `SRCalculator` protocol
- Implement: `calculate(data, **params)`, `get_default_parameters()`, `get_parameter_info()`
- Use `@registry.register_sr_method("method_name")` decorator
- Return `SRResult` with ticker, data, levels, method, parameters, calculation_time

Example structure for external methods:
```python
@registry.register_sr_method("custom_method")
class CustomCalculator(BaseSRCalculator):
    def __init__(self):
        super().__init__(SRMethod.PIVOT_POINTS)  # Use any enum value

    def calculate(self, data: pd.DataFrame, **params) -> SRResult:
        # Implementation
        pass
```

## Storage Organization

Results are stored in a hierarchical structure:
```
{CONNORS_HOME}/sr_calculations/
├── pivot_points/
│   ├── {ticker}_{market}_{start}_{end}.json
│   └── plots/
│       └── {ticker}_{market}_{start}_{end}.html
├── fractal/
├── vwap_zones/
└── volume_profile/
```

## Testing Architecture

Tests are organized by component:
- `TestSRCalculators`: Unit tests for all built-in calculators
- `TestSRService`: Service layer tests with mocked data sources
- `TestSRIntegration`: End-to-end workflow tests

All tests use `setUp()` to create realistic OHLCV test data with proper pandas DatetimeIndex.

## Code Standards

- **Python Version**: 3.13+
- **Formatting**: Black (line length 88), configured in pyproject.toml
- **Import Sorting**: isort with black profile
- **Type Checking**: mypy with `check_untyped_defs=true`
- **Documentation**: Comprehensive docstrings for all public APIs
- **Column Naming**: SR calculators expect title-case OHLCV columns (Open, High, Low, Close, Volume)
