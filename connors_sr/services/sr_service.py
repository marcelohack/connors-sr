"""
Support & Resistance Service

Provides high-level interface for Support & Resistance calculation operations,
integrating with data sources, calculators, and file storage.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import connors_datafetch.datasources.finnhub  # noqa: F401
import connors_datafetch.datasources.fmp  # noqa: F401
import connors_datafetch.datasources.polygon  # noqa: F401

# Import all datasources to ensure registration
import connors_datafetch.datasources.yfinance  # noqa: F401
import pandas as pd
import plotly.graph_objects as go
from connors_datafetch.config.manager import DataFetchConfigManager
from connors_datafetch.core.timespan import TimespanCalculator
from connors_datafetch.services.datafetch_service import DataFetchService
from plotly.subplots import make_subplots

from connors_sr.core.registry import registry
from connors_sr.core.support_resistance import (
    FractalCalculator,
    PivotPointsCalculator,
    SRLevel,
    SRMethod,
    SRResult,
    VolumeProfileCalculator,
    VWAPZonesCalculator,
)
from connors_sr.services.base import BaseService


@dataclass
class SRCalculationRequest:
    """Request for SR calculation"""

    ticker: str
    method: Union[SRMethod, str]
    parameters: Optional[Dict[str, Any]] = None
    datasource: str = "yfinance"
    dataset_file: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    interval: str = "1d"
    market_config: str = "america"
    timeframe: Optional[str] = None
    save_results: bool = False
    save_plot: bool = False
    show_plot: bool = False


@dataclass
class SRServiceResult:
    """Container for SR service results"""

    ticker: str
    method: Union[SRMethod, str]
    results: Optional[SRResult]
    plot_path: Optional[str] = None
    results_path: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class SRService(BaseService):
    """Service for Support & Resistance calculations"""

    def __init__(self) -> None:
        super().__init__()
        self.registry = registry
        self.config_manager = DataFetchConfigManager()
        self.download_service = DataFetchService()
        self.timespan_calculator = TimespanCalculator()

        # Initialize calculators - dict accepts both SRMethod and str keys
        self.calculators: Dict[Union[SRMethod, str], Any] = {
            SRMethod.PIVOT_POINTS: PivotPointsCalculator(),
            SRMethod.FRACTAL: FractalCalculator(),
            SRMethod.VWAP_ZONES: VWAPZonesCalculator(),
            SRMethod.VOLUME_PROFILE: VolumeProfileCalculator(),
        }

        # Ensure CONNORS_HOME directory structure
        self.connors_home = Path(
            os.environ.get("CONNORS_HOME", Path.home() / ".connors")
        )
        self.sr_base_dir = self.connors_home / "sr_calculations"
        self._ensure_directory_exists(self.sr_base_dir)

    def get_available_methods(self) -> List[str]:
        """Get list of available SR calculation methods"""
        return [method.value for method in SRMethod]

    def get_method_info(self, method: Union[SRMethod, str]) -> Dict[str, Any]:
        """Get information about a specific SR method"""
        try:
            if isinstance(method, str):
                method = SRMethod(method)
        except ValueError:
            # Invalid method name
            return {}

        calculator = self.calculators.get(method)
        if not calculator:
            return {}

        return {
            "name": method.value,
            "description": calculator.__class__.__doc__
            or f"{method.value.title()} calculator",
            "default_parameters": calculator.get_default_parameters(),
            "parameter_info": calculator.get_parameter_info(),
        }

    def get_all_methods_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available SR methods"""
        return {method.value: self.get_method_info(method) for method in SRMethod}

    def calculate_sr(self, request: SRCalculationRequest) -> SRServiceResult:
        """
        Calculate Support & Resistance levels for a given ticker and method

        Args:
            request: SRCalculationRequest containing all parameters

        Returns:
            SRServiceResult with calculation results and file paths
        """
        try:
            # Convert string method to enum for built-in methods, keep as string for external
            method: Union[SRMethod, str]
            if isinstance(request.method, str):
                # Check if it's an external method first
                if request.method in self.calculators:
                    method = request.method  # Keep as string for external methods
                else:
                    # Try to convert to built-in SRMethod enum
                    try:
                        method = SRMethod(request.method.lower())
                    except ValueError:
                        # If not a valid enum, keep as string (might be external)
                        method = request.method
            else:
                method = request.method

            # Get or download data
            if request.dataset_file:
                data = self._load_dataset_file(request.dataset_file, request.ticker)
            else:
                data = self._download_data(request)

            # Get calculator
            calculator = self.calculators.get(method)
            if not calculator:
                raise ValueError(f"Calculator not found for method: {method}")

            # Prepare parameters
            calc_params = calculator.get_default_parameters()
            if request.parameters:
                calc_params.update(request.parameters)

            # Calculate SR levels
            sr_result = calculator.calculate(data, **calc_params)

            # Save results if requested
            results_path = None
            if request.save_results:
                results_path = self._save_results(sr_result, request)

            # Generate and save plot if requested
            plot_path = None
            if request.save_plot or request.show_plot:
                plot_path = self._generate_plot(sr_result, request)
                if request.show_plot:
                    self._show_plot(plot_path)

            return SRServiceResult(
                ticker=request.ticker,
                method=method,
                results=sr_result,
                plot_path=plot_path,
                results_path=results_path,
                success=sr_result.success,
                error=sr_result.error,
            )

        except Exception as e:
            self.logger.error(f"SR calculation failed for {request.ticker}: {e}")
            return SRServiceResult(
                ticker=request.ticker,
                method=method if "method" in locals() else SRMethod.PIVOT_POINTS,
                results=None,
                success=False,
                error=str(e),
            )

    def calculate_multiple_methods(
        self, ticker: str, methods: List[Union[SRMethod, str]], **common_params
    ) -> Dict[str, SRServiceResult]:
        """
        Calculate SR levels using multiple methods for comparison

        Args:
            ticker: Stock ticker
            methods: List of methods to calculate
            **common_params: Common parameters for all calculations

        Returns:
            Dictionary mapping method names to results
        """
        results = {}

        for method in methods:
            method_name = method.value if isinstance(method, SRMethod) else method

            request = SRCalculationRequest(
                ticker=ticker, method=method, **common_params
            )

            result = self.calculate_sr(request)
            results[method_name] = result

        return results

    def _download_data(self, request: SRCalculationRequest) -> pd.DataFrame:
        """Download data using the download service"""
        # Calculate date range
        if request.timeframe:
            date_result = self.download_service.calculate_dates_from_timeframe(
                timeframe=request.timeframe,
                start_date=request.start,
                end_date=request.end,
            )
            start_date = date_result["start"]
            end_date = date_result["end"]
        else:
            defaults = self.download_service.get_default_dates()
            start_date = request.start or defaults["start"]
            end_date = request.end or defaults["end"]

        # Download data
        download_result = self.download_service.download_data(
            datasource=request.datasource,
            ticker=request.ticker,
            start=start_date,
            end=end_date,
            interval=request.interval,
            market=request.market_config,
            timeframe=request.timeframe,
        )

        if not download_result.success:
            raise ValueError(f"Data download failed: {download_result.error}")

        # Add ticker attribute to dataframe for reference
        if download_result.data is not None:
            download_result.data.ticker = request.ticker  # type: ignore[attr-defined]

            # Normalize column names for SR calculator (expects title case)
            data = self._prepare_dataframe_for_sr_calculation(download_result.data)
            data.ticker = request.ticker  # type: ignore[attr-defined]
            return data
        else:
            raise ValueError("Downloaded data is None")

    def _load_dataset_file(self, dataset_file: str, ticker: str) -> pd.DataFrame:
        """Load data from a dataset file"""
        file_path = Path(dataset_file)

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        # Load data based on file extension
        if file_path.suffix.lower() == ".csv":
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        elif file_path.suffix.lower() == ".json":
            # Try different JSON loading approaches
            try:
                # First try standard JSON loading (for OHLCV data)
                data = pd.read_json(file_path)

                # If the result is a dict-like structure, try converting to DataFrame
                if hasattr(data, "keys") and not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data)

                # Ensure we have a proper datetime index
                if not isinstance(data.index, pd.DatetimeIndex):
                    if "date" in data.columns:
                        data.set_index("date", inplace=True)
                        data.index = pd.to_datetime(data.index)
                    elif data.index.name in ["date", "Date", "datetime"]:
                        data.index = pd.to_datetime(data.index)
                    else:
                        data.index = pd.to_datetime(data.index)

            except Exception as e:
                # Fallback: try orient='index' approach
                try:
                    data = pd.read_json(file_path, orient="index")
                    data.index = pd.to_datetime(data.index)
                except Exception as e2:
                    raise ValueError(
                        f"Could not load JSON file: {e}. Fallback attempt also failed: {e2}"
                    )
        else:
            raise ValueError(f"Unsupported dataset file format: {file_path.suffix}")

        # Validate required columns (check both lowercase and title case)
        required_columns_title = ["Open", "High", "Low", "Close", "Volume"]
        required_columns_lower = ["open", "high", "low", "close", "volume"]

        # Check if we have title case columns
        title_case_available = all(
            col in data.columns for col in required_columns_title
        )
        # Check if we have lowercase columns
        lower_case_available = all(
            col in data.columns for col in required_columns_lower
        )

        if not title_case_available and not lower_case_available:
            # Show what columns we actually have
            available_cols = list(data.columns)
            raise ValueError(
                f"Dataset file missing required columns. "
                f"Need either {required_columns_title} or {required_columns_lower}. "
                f"Available columns: {available_cols}"
            )

        # Add ticker attribute
        data.ticker = ticker

        # Normalize column names for SR calculator (only if needed)
        if lower_case_available and not title_case_available:
            # Convert lowercase to title case
            data = self._prepare_dataframe_for_sr_calculation(data)
        # If title case is already available, use as-is

        data.ticker = ticker
        return data

    def _prepare_dataframe_for_sr_calculation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame with lowercase columns to title case format required by SR calculators

        This follows the same pattern as the backtest service, which normalizes data after
        fetching from datasources to ensure consistent column naming.
        """
        df_sr = df.copy()

        # Map lowercase column names to title case format expected by SR calculators
        column_mapping = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }

        # Rename columns to title case
        df_sr = df_sr.rename(columns=column_mapping)

        return df_sr

    def _save_results(self, sr_result: SRResult, request: SRCalculationRequest) -> str:
        """Save SR calculation results to file"""
        # Create organized directory structure
        # Use request method name for external methods, otherwise use enum value
        if isinstance(request.method, str):
            # External method - use the method name directly
            method_folder_name = request.method
        else:
            # Built-in method - use enum value
            method_folder_name = sr_result.method.value

        method_dir = self.sr_base_dir / method_folder_name
        self._ensure_directory_exists(method_dir)

        # Generate filename with new pattern: {ticker}_{market}_{start}_{end}.json
        if hasattr(sr_result.data, "index") and not sr_result.data.empty:
            start_date = sr_result.data.index.min().strftime("%Y-%m-%d")
            end_date = sr_result.data.index.max().strftime("%Y-%m-%d")
        else:
            start_date = "unknown"
            end_date = "unknown"

        filename = (
            f"{request.ticker}_{request.market_config}_{start_date}_{end_date}.json"
        )
        results_path = method_dir / filename

        # Prepare results data
        results_data = {
            "ticker": sr_result.ticker,
            "method": method_folder_name,  # Use the same method name as folder
            "calculation_time": sr_result.calculation_time,
            "parameters": sr_result.parameters,
            "levels": [
                {
                    "level": level.level,
                    "level_type": level.level_type,
                    "method": level.method.value,
                    "strength": level.strength,
                    "touches": level.touches,
                    "first_occurrence": level.first_occurrence.isoformat(),
                    "last_occurrence": level.last_occurrence.isoformat(),
                    "metadata": level.metadata,
                }
                for level in sr_result.levels
            ],
            "data_shape": sr_result.data.shape,
            "data_columns": list(sr_result.data.columns),
            "date_range": {"start": start_date, "end": end_date},
            "calculated_at": datetime.now().isoformat(),
        }

        # Save to JSON file
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        self.logger.info(f"SR results saved to: {results_path}")
        return str(results_path)

    def _generate_plot(self, sr_result: SRResult, request: SRCalculationRequest) -> str:
        """Generate interactive plot of SR levels"""
        # Create organized directory structure for plots
        # Use request method name for external methods, otherwise use enum value
        if isinstance(request.method, str):
            # External method - use the method name directly
            method_folder_name = request.method
        else:
            # Built-in method - use enum value
            method_folder_name = sr_result.method.value

        plots_dir = self.sr_base_dir / method_folder_name / "plots"
        self._ensure_directory_exists(plots_dir)

        # Generate filename with new pattern: {ticker}_{market}_{start}_{end}.html
        if hasattr(sr_result.data, "index") and not sr_result.data.empty:
            start_date = sr_result.data.index.min().strftime("%Y-%m-%d")
            end_date = sr_result.data.index.max().strftime("%Y-%m-%d")
        else:
            start_date = "unknown"
            end_date = "unknown"

        filename = (
            f"{request.ticker}_{request.market_config}_{start_date}_{end_date}.html"
        )
        plot_path = plots_dir / filename

        # Determine method name for plot title
        if isinstance(request.method, str):
            # External method - use the method name directly
            method_display_name = request.method.replace("_", " ").title()
        else:
            # Built-in method - use enum value
            method_display_name = sr_result.method.value.replace("_", " ").title()

        # Create subplot with secondary y-axis for volume
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                f"{request.ticker} - {method_display_name} Support & Resistance",
                "Volume",
            ),
            row_width=[0.7, 0.3],
        )

        # Add OHLC candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=sr_result.data.index,
                open=sr_result.data["Open"],
                high=sr_result.data["High"],
                low=sr_result.data["Low"],
                close=sr_result.data["Close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        # Add volume bars
        colors = [
            "red" if close < open else "green"
            for close, open in zip(sr_result.data["Close"], sr_result.data["Open"])
        ]

        fig.add_trace(
            go.Bar(
                x=sr_result.data.index,
                y=sr_result.data["Volume"],
                marker_color=colors,
                name="Volume",
                opacity=0.6,
            ),
            row=2,
            col=1,
        )

        # Add support and resistance levels
        support_levels = [
            level for level in sr_result.levels if level.level_type == "support"
        ]
        resistance_levels = [
            level for level in sr_result.levels if level.level_type == "resistance"
        ]

        # Get price range for level filtering
        price_min = sr_result.data["Low"].min()
        price_max = sr_result.data["High"].max()
        price_range = price_max - price_min

        # Filter levels that are within a reasonable range of actual prices
        # More generous filtering for resistance levels (can be further above current prices)
        support_margin = price_range * 0.5  # 50% below for support
        resistance_margin = (
            price_range * 1.0
        )  # 100% above for resistance (more generous)

        # Filter and sort levels by strength and touches
        filtered_support_levels = [
            level
            for level in support_levels
            if price_min - support_margin <= level.level <= price_max + support_margin
        ]
        filtered_resistance_levels = [
            level
            for level in resistance_levels
            if price_min - resistance_margin
            <= level.level
            <= price_max + resistance_margin
        ]

        # Sort support levels by strength and touches
        valid_support_levels = sorted(
            filtered_support_levels, key=lambda x: (x.strength, x.touches), reverse=True
        )[
            :4
        ]  # Top 4 levels

        # For resistance, sort by level value (highest first) to show overhead resistance
        valid_resistance_levels = sorted(
            filtered_resistance_levels, key=lambda x: x.level, reverse=True
        )[
            :4
        ]  # Top 4 highest levels

        # Add support levels (green) - more prominent styling
        for i, level in enumerate(valid_support_levels):
            line_width = max(3 - i * 0.2, 1.5)  # Thicker lines for stronger levels
            fig.add_hline(
                y=level.level,
                line_dash="solid",
                line_color="rgba(0, 128, 0, 0.9)",  # Solid green with high opacity
                line_width=line_width,
                annotation_text=f"S: ${level.level:.2f} (Str: {level.strength:.2f})",
                annotation_position="bottom right" if i % 2 == 0 else "top right",
                annotation=dict(
                    bgcolor="rgba(0, 128, 0, 0.8)",
                    bordercolor="green",
                    borderwidth=1,
                    font=dict(color="white", size=10),
                ),
                row=1,
                col=1,
            )

        # Add resistance levels (red) - more prominent styling
        for i, level in enumerate(valid_resistance_levels):
            line_width = max(3 - i * 0.2, 1.5)  # Thicker lines for stronger levels
            fig.add_hline(
                y=level.level,
                line_dash="solid",
                line_color="rgba(255, 0, 0, 0.9)",  # Solid red with high opacity
                line_width=line_width,
                annotation_text=f"R: ${level.level:.2f} (Str: {level.strength:.2f})",
                annotation_position="top right" if i % 2 == 0 else "bottom right",
                annotation=dict(
                    bgcolor="rgba(255, 0, 0, 0.8)",
                    bordercolor="red",
                    borderwidth=1,
                    font=dict(color="white", size=10),
                ),
                row=1,
                col=1,
            )

        # Update layout
        fig.update_layout(
            title=f"{request.ticker} - {method_display_name} Support & Resistance Analysis",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template="plotly_white",
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        # Save plot
        fig.write_html(plot_path)

        self.logger.info(f"SR plot saved to: {plot_path}")
        return str(plot_path)

    def _show_plot(self, plot_path: str) -> None:
        """Show plot in browser"""
        import webbrowser

        webbrowser.open(f"file://{plot_path}")

    def list_saved_results(
        self, method: Optional[str] = None, market: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """List saved SR calculation results"""
        results = []

        if method:
            method_dirs = [self.sr_base_dir / method]
        else:
            method_dirs = [d for d in self.sr_base_dir.iterdir() if d.is_dir()]

        for method_dir in method_dirs:
            if not method_dir.is_dir():
                continue

            if market:
                market_dirs = [method_dir / market]
            else:
                market_dirs = [d for d in method_dir.iterdir() if d.is_dir()]

            for market_dir in market_dirs:
                if not market_dir.is_dir() or market_dir.name == "plots":
                    continue

                for result_file in market_dir.glob("*.json"):
                    try:
                        # Parse filename to extract info
                        name_parts = result_file.stem.split("_")
                        if len(name_parts) >= 3:
                            ticker = name_parts[0]
                            start_date = name_parts[1]
                            end_date = name_parts[2]
                        else:
                            ticker = result_file.stem
                            start_date = "unknown"
                            end_date = "unknown"

                        results.append(
                            {
                                "ticker": ticker,
                                "method": method_dir.name,
                                "market": market_dir.name,
                                "start_date": start_date,
                                "end_date": end_date,
                                "file_path": str(result_file),
                                "modified": datetime.fromtimestamp(
                                    result_file.stat().st_mtime
                                ).isoformat(),
                            }
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Could not parse result file {result_file}: {e}"
                        )

        # Sort by modification time (newest first)
        results.sort(key=lambda x: x["modified"], reverse=True)
        return results

    def load_saved_result(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load a saved SR calculation result"""
        try:
            with open(file_path, "r") as f:
                result: Dict[str, Any] = json.load(f)
                return result
        except Exception as e:
            self.logger.error(f"Failed to load saved result from {file_path}: {e}")
            return None

    def delete_saved_result(self, file_path: str) -> bool:
        """Delete a saved SR calculation result"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()

                # Also delete corresponding plot if it exists
                plot_path = (
                    path.parent.parent
                    / "plots"
                    / path.parent.name
                    / (path.stem + ".html")
                )
                if plot_path.exists():
                    plot_path.unlink()

                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete saved result {file_path}: {e}")
            return False

    # Utility methods for CLI and UI integration
    def get_datasources(self) -> List[str]:
        """Get available datasources"""
        return self.download_service.get_datasources()

    def get_market_configs(self) -> List[str]:
        """Get available market configurations"""
        return self.download_service.get_market_configs()

    def get_available_timeframes(self) -> List[str]:
        """Get available timeframes"""
        return self.download_service.get_available_timeframes()

    def get_market_config_info(self, config: str) -> Optional[Dict[str, Any]]:
        """Get market configuration info"""
        return self.download_service.get_market_config_info(config)

    def calculate_dates_from_timeframe(
        self,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, str]:
        """Calculate dates from timeframe"""
        return self.download_service.calculate_dates_from_timeframe(
            timeframe, start_date, end_date
        )

    def get_timeframe_description(self, timeframe: str) -> str:
        """Get timeframe description"""
        return self.download_service.get_timeframe_description(timeframe)

    def get_default_dates(self) -> Dict[str, str]:
        """Get default date range"""
        return self.download_service.get_default_dates()

    def str2bool(self, v: Union[bool, str]) -> bool:
        """Convert string to boolean (for CLI compatibility)"""
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise ValueError(f"Boolean value expected, got: {v}")

    def load_external_method(self, file_path: str) -> str:
        """
        Load an external SR calculation method from a Python file and register it

        Args:
            file_path: Path to the Python file containing the SR method

        Returns:
            The name the method was registered under
        """
        import importlib.util
        import sys
        from pathlib import Path

        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"SR method file not found: {file_path_obj}")

            if not file_path_obj.suffix == ".py":
                raise ValueError("SR method file must be a Python (.py) file")

            # Create module spec and load the module
            module_name = (
                f"external_sr_method_{file_path_obj.stem}_{hash(str(file_path_obj))}"
            )
            spec = importlib.util.spec_from_file_location(module_name, file_path_obj)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module from {file_path_obj}")

            module = importlib.util.module_from_spec(spec)

            # Add the module to sys.modules to ensure proper import context
            sys.modules[module_name] = module

            # Ensure the registry is available in the module's namespace
            module.__dict__["registry"] = self.registry

            # Also make sure common imports are available
            try:
                import numpy as np
                import pandas as pd
                from connors.core.support_resistance import (
                    BaseSRCalculator,
                    SRLevel,
                    SRMethod,
                    SRResult,
                )

                module.__dict__["pd"] = pd
                module.__dict__["pandas"] = pd
                module.__dict__["np"] = np
                module.__dict__["numpy"] = np
                module.__dict__["BaseSRCalculator"] = BaseSRCalculator
                module.__dict__["SRResult"] = SRResult
                module.__dict__["SRLevel"] = SRLevel
                module.__dict__["SRMethod"] = SRMethod
            except ImportError:
                pass

            try:
                import talib

                module.__dict__["talib"] = talib
            except ImportError:
                pass

            try:
                import pandas_ta as ta

                module.__dict__["ta"] = ta
                module.__dict__["pandas_ta"] = ta
            except ImportError:
                pass

            # Store current SR methods to detect newly registered ones
            existing_methods = set(self.registry._sr_methods.keys())

            # Execute the module (this will run the @registry.register_sr_method decorators)
            try:
                spec.loader.exec_module(module)
            except Exception as exec_error:
                # Provide more detailed error information
                raise ImportError(f"Failed to execute module: {exec_error}")

            # Find newly registered methods
            new_methods = set(self.registry._sr_methods.keys()) - existing_methods

            # If no methods were registered via decorator, try to find and register SR method classes manually
            if not new_methods:
                method_classes = []
                for attr_name in dir(module):
                    try:
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type)
                            and hasattr(attr, "__bases__")
                            and any(
                                "SRCalculator" in str(base) for base in attr.__bases__
                            )
                        ):
                            method_classes.append((attr_name, attr))
                    except Exception:
                        continue

                if not method_classes:
                    raise ValueError(
                        "No SR method classes found in the module. "
                        "Make sure your method class inherits from BaseSRCalculator or implements SRCalculator protocol"
                    )

                # Use the class name as the method name
                if len(method_classes) > 1:
                    raise ValueError(
                        f"Multiple SR method classes found. Classes found: {[name for name, _ in method_classes]}. "
                        f"Please ensure only one SR method class is defined in the file."
                    )

                class_name, class_obj = method_classes[0]
                method_name = class_name.lower()
                self.registry._sr_methods[method_name] = class_obj
                class_obj._registry_name = method_name  # type: ignore[attr-defined]

                # Add the method instance to available calculators
                self.calculators[method_name] = class_obj()
                return method_name
            else:
                # Method was registered via decorator - instantiate and add to calculators
                registered_method_name = list(new_methods)[0]
                method_class = self.registry._sr_methods[registered_method_name]
                self.calculators[registered_method_name] = method_class()
                return registered_method_name

        except Exception as e:
            self.logger.error(
                f"Failed to load external SR method from {file_path}: {e}"
            )
            raise
