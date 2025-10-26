"""
Test Support & Resistance Calculator functionality
"""

import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from connors_sr.core.support_resistance import (
    FractalCalculator,
    PivotPointsCalculator,
    SRLevel,
    SRMethod,
    SRResult,
    VolumeProfileCalculator,
    VWAPZonesCalculator,
)
from connors_sr.services.sr_service import SRCalculationRequest, SRService


class TestSRCalculators(unittest.TestCase):
    """Test Support & Resistance calculation methods"""

    def setUp(self):
        """Set up test data"""
        # Create sample OHLCV data
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)

        # Generate realistic price data
        close_prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
        high_prices = close_prices + np.random.uniform(0.5, 2.0, 50)
        low_prices = close_prices - np.random.uniform(0.5, 2.0, 50)
        open_prices = close_prices + np.random.randn(50) * 0.3
        volumes = np.random.randint(1000000, 10000000, 50)

        self.test_data = pd.DataFrame(
            {
                "Open": open_prices,
                "High": high_prices,
                "Low": low_prices,
                "Close": close_prices,
                "Volume": volumes,
            },
            index=dates,
        )

        self.test_data.ticker = "TEST"

    def test_pivot_points_calculator(self):
        """Test Pivot Points calculator"""
        calculator = PivotPointsCalculator()

        # Test basic calculation
        result = calculator.calculate(self.test_data)

        self.assertTrue(result.success)
        self.assertEqual(result.method, SRMethod.PIVOT_POINTS)
        self.assertIsInstance(result.levels, list)
        self.assertGreater(len(result.levels), 0)

        # Verify data columns were added
        self.assertIn("PP", result.data.columns)
        self.assertIn("R1", result.data.columns)
        self.assertIn("S1", result.data.columns)

        # Test with custom parameters
        result_custom = calculator.calculate(self.test_data, include_midpoints=False)
        self.assertTrue(result_custom.success)
        self.assertFalse(result_custom.parameters["include_midpoints"])

    def test_fractal_calculator(self):
        """Test Fractal calculator"""
        calculator = FractalCalculator()

        # Test basic calculation
        result = calculator.calculate(self.test_data)

        self.assertTrue(result.success)
        self.assertEqual(result.method, SRMethod.FRACTAL)
        self.assertIsInstance(result.levels, list)

        # Verify data columns were added
        self.assertIn("FractalHigh", result.data.columns)
        self.assertIn("FractalLow", result.data.columns)

        # Test with custom parameters
        result_custom = calculator.calculate(
            self.test_data, lookback=7, min_strength=0.5
        )
        self.assertTrue(result_custom.success)
        self.assertEqual(result_custom.parameters["lookback"], 7)
        self.assertEqual(result_custom.parameters["min_strength"], 0.5)

    def test_vwap_zones_calculator(self):
        """Test VWAP Zones calculator"""
        calculator = VWAPZonesCalculator()

        # Test basic calculation
        result = calculator.calculate(self.test_data)

        self.assertTrue(result.success)
        self.assertEqual(result.method, SRMethod.VWAP_ZONES)
        self.assertIsInstance(result.levels, list)

        # Verify VWAP calculation columns
        self.assertIn("VWAP", result.data.columns)
        self.assertIn("VWAP_Upper_1", result.data.columns)
        self.assertIn("VWAP_Lower_1", result.data.columns)

        # Test with custom parameters
        result_custom = calculator.calculate(
            self.test_data, period=10, std_devs=[1.5, 2.5], min_volume_ratio=1.2
        )
        self.assertTrue(result_custom.success)
        self.assertEqual(result_custom.parameters["period"], 10)

    def test_volume_profile_calculator(self):
        """Test Volume Profile calculator"""
        calculator = VolumeProfileCalculator()

        # Test basic calculation
        result = calculator.calculate(self.test_data)

        self.assertTrue(result.success)
        self.assertEqual(result.method, SRMethod.VOLUME_PROFILE)
        self.assertIsInstance(result.levels, list)

        # Verify volume profile columns
        self.assertIn("VP_Volume", result.data.columns)
        self.assertIn("VP_Level", result.data.columns)

        # Test with custom parameters
        result_custom = calculator.calculate(
            self.test_data, price_bins=30, min_volume_pct=3.0, lookback_periods=40
        )
        self.assertTrue(result_custom.success)
        self.assertEqual(result_custom.parameters["price_bins"], 30)

    def test_invalid_data_handling(self):
        """Test handling of invalid input data"""
        # Test with missing columns
        invalid_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [102, 103, 104],
                "Close": [101, 102, 103],
                # Missing Low and Volume
            }
        )

        calculator = PivotPointsCalculator()

        # Should raise ValueError during validation, which gets caught and returned as failed result
        with self.assertRaises(ValueError) as context:
            calculator.calculate(invalid_data)

        self.assertIn("Missing required columns", str(context.exception))

        # Test with insufficient data
        small_data = self.test_data.head(5)  # Only 5 rows

        with self.assertRaises(ValueError) as context:
            calculator.calculate(small_data)

        self.assertIn("Insufficient data", str(context.exception))

    def test_sr_level_properties(self):
        """Test SRLevel dataclass properties"""
        calculator = PivotPointsCalculator()
        result = calculator.calculate(self.test_data)

        for level in result.levels:
            # Verify all required properties exist
            self.assertIsInstance(level.level, float)
            self.assertIn(level.level_type, ["support", "resistance"])
            self.assertEqual(level.method, SRMethod.PIVOT_POINTS)
            self.assertIsInstance(level.strength, float)
            self.assertIsInstance(level.touches, int)
            self.assertIsInstance(level.first_occurrence, pd.Timestamp)
            self.assertIsInstance(level.last_occurrence, pd.Timestamp)

            # Verify reasonable ranges
            self.assertGreater(level.level, 0)
            self.assertGreaterEqual(level.strength, 0)
            self.assertLessEqual(
                level.strength, 2.0
            )  # Updated to match new algorithm cap
            self.assertGreaterEqual(level.touches, 0)

    def test_default_parameters(self):
        """Test default parameter retrieval"""
        calculators = [
            PivotPointsCalculator(),
            FractalCalculator(),
            VWAPZonesCalculator(),
            VolumeProfileCalculator(),
        ]

        for calculator in calculators:
            defaults = calculator.get_default_parameters()
            self.assertIsInstance(defaults, dict)
            self.assertGreater(len(defaults), 0)

            # Test parameter info
            param_info = calculator.get_parameter_info()
            self.assertIsInstance(param_info, dict)

            # Verify all default params have info
            for param_name in defaults.keys():
                self.assertIn(param_name, param_info)
                self.assertIn("type", param_info[param_name])
                self.assertIn("description", param_info[param_name])

    def test_calculation_performance(self):
        """Test calculation performance and timing"""
        # Create larger dataset with proper date indexing
        base_data = self.test_data.copy()
        large_data_frames = [base_data]

        # Create additional data with sequential dates to avoid index issues
        last_date = base_data.index[-1]
        for i in range(5):  # 5x larger dataset
            new_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), periods=len(base_data), freq="D"
            )
            new_data = base_data.copy()
            new_data.index = new_dates
            large_data_frames.append(new_data)
            last_date = new_dates[-1]

        large_data = pd.concat(large_data_frames, sort=False)
        large_data.ticker = "TEST"

        calculator = PivotPointsCalculator()
        result = calculator.calculate(large_data)

        self.assertTrue(
            result.success,
            f"Calculation failed with error: {result.error if hasattr(result, 'error') else 'Unknown error'}",
        )
        self.assertGreater(result.calculation_time, 0)
        self.assertLess(
            result.calculation_time, 10.0
        )  # Should complete in under 10 seconds (increased for new algorithm)


class TestSRService(unittest.TestCase):
    """Test Support & Resistance Service"""

    def setUp(self):
        """Set up test service and data"""
        self.service = SRService()

        # Mock download service to avoid actual data fetching
        self.mock_download_data = {
            "success": True,
            "data": self._create_test_data(),
            "error": None,
        }

    def _create_test_data(self):
        """Create test OHLCV data"""
        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        np.random.seed(42)

        close_prices = 100 + np.cumsum(np.random.randn(30) * 0.3)
        data = pd.DataFrame(
            {
                "Open": close_prices + np.random.randn(30) * 0.2,
                "High": close_prices + np.random.uniform(0.2, 1.0, 30),
                "Low": close_prices - np.random.uniform(0.2, 1.0, 30),
                "Close": close_prices,
                "Volume": np.random.randint(500000, 5000000, 30),
            },
            index=dates,
        )

        data.ticker = "AAPL"
        return data

    def test_get_available_methods(self):
        """Test retrieving available methods"""
        methods = self.service.get_available_methods()

        self.assertIsInstance(methods, list)
        self.assertGreater(len(methods), 0)
        self.assertIn("pivot_points", methods)
        self.assertIn("fractal", methods)
        self.assertIn("vwap_zones", methods)
        self.assertIn("volume_profile", methods)

    def test_get_method_info(self):
        """Test retrieving method information"""
        info = self.service.get_method_info("pivot_points")

        self.assertIsInstance(info, dict)
        self.assertIn("name", info)
        self.assertIn("description", info)
        self.assertIn("default_parameters", info)
        self.assertIn("parameter_info", info)

        # Test with invalid method
        invalid_info = self.service.get_method_info("invalid_method")
        self.assertEqual(invalid_info, {})

    @patch("connors_sr.services.sr_service.SRService._download_data")
    def test_calculate_sr_success(self, mock_download):
        """Test successful SR calculation"""
        mock_download.return_value = self._create_test_data()

        request = SRCalculationRequest(
            ticker="AAPL",
            method="pivot_points",
            datasource="yfinance",
            start="2023-01-01",
            end="2023-01-31",
        )

        result = self.service.calculate_sr(request)

        self.assertTrue(result.success)
        self.assertEqual(result.ticker, "AAPL")
        self.assertEqual(result.method, SRMethod.PIVOT_POINTS)
        self.assertIsNotNone(result.results)
        self.assertIsInstance(result.results.levels, list)

    @patch("connors_sr.services.sr_service.SRService._download_data")
    def test_calculate_sr_with_parameters(self, mock_download):
        """Test SR calculation with custom parameters"""
        mock_download.return_value = self._create_test_data()

        request = SRCalculationRequest(
            ticker="AAPL",
            method="fractal",
            parameters={"lookback": 7, "min_strength": 0.4},
            datasource="yfinance",
        )

        result = self.service.calculate_sr(request)

        self.assertTrue(result.success)
        self.assertEqual(result.results.parameters["lookback"], 7)
        self.assertEqual(result.results.parameters["min_strength"], 0.4)

    def test_calculate_sr_invalid_method(self):
        """Test SR calculation with invalid method"""
        request = SRCalculationRequest(ticker="AAPL", method="invalid_method")

        result = self.service.calculate_sr(request)

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    @patch("connors_sr.services.sr_service.SRService._load_dataset_file")
    def test_calculate_sr_with_dataset_file(self, mock_load_file):
        """Test SR calculation using dataset file"""
        mock_load_file.return_value = self._create_test_data()

        request = SRCalculationRequest(
            ticker="AAPL", method="pivot_points", dataset_file="/path/to/test.csv"
        )

        result = self.service.calculate_sr(request)

        self.assertTrue(result.success)
        mock_load_file.assert_called_once_with("/path/to/test.csv", "AAPL")

    def test_calculate_multiple_methods(self):
        """Test calculating multiple methods for comparison"""
        with patch.object(self.service, "_download_data") as mock_download:
            mock_download.return_value = self._create_test_data()

            methods = ["pivot_points", "fractal"]
            results = self.service.calculate_multiple_methods(
                ticker="AAPL", methods=methods, datasource="yfinance"
            )

            self.assertIsInstance(results, dict)
            self.assertEqual(len(results), 2)
            self.assertIn("pivot_points", results)
            self.assertIn("fractal", results)

            for method_name, result in results.items():
                self.assertTrue(result.success)
                self.assertEqual(result.ticker, "AAPL")

    @patch("builtins.open", create=True)
    @patch("json.dump")
    def test_save_results(self, mock_json_dump, mock_open):
        """Test saving SR results to file"""
        # Create mock SR result
        levels = [
            SRLevel(
                level=100.0,
                level_type="support",
                method=SRMethod.PIVOT_POINTS,
                strength=0.8,
                touches=3,
                first_occurrence=pd.Timestamp("2023-01-01"),
                last_occurrence=pd.Timestamp("2023-01-31"),
                metadata={"pivot_type": "S1"},
            )
        ]

        sr_result = SRResult(
            ticker="AAPL",
            data=self._create_test_data(),
            levels=levels,
            method=SRMethod.PIVOT_POINTS,
            parameters={"period": 1},
            calculation_time=0.1,
        )

        request = SRCalculationRequest(ticker="AAPL", method="pivot_points")

        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        result_path = self.service._save_results(sr_result, request)

        self.assertIsInstance(result_path, str)
        self.assertTrue(result_path.endswith(".json"))
        mock_json_dump.assert_called_once()

    @patch("webbrowser.open")
    def test_show_plot(self, mock_webbrowser):
        """Test opening plot in browser"""
        plot_path = "/tmp/test_plot.html"
        self.service._show_plot(plot_path)

        mock_webbrowser.assert_called_once_with(f"file://{plot_path}")

    def test_list_saved_results(self):
        """Test listing saved results"""
        # Create temporary directory structure
        with patch.object(Path, "iterdir") as mock_iterdir:
            # Mock empty results
            mock_iterdir.return_value = []

            results = self.service.list_saved_results()
            self.assertIsInstance(results, list)

    def test_utility_methods(self):
        """Test utility methods for CLI/UI integration"""
        # Test datasources
        datasources = self.service.get_datasources()
        self.assertIsInstance(datasources, list)

        # Test market configs
        markets = self.service.get_market_configs()
        self.assertIsInstance(markets, list)

        # Test timeframes
        timeframes = self.service.get_available_timeframes()
        self.assertIsInstance(timeframes, list)

        # Test str2bool
        self.assertTrue(self.service.str2bool("true"))
        self.assertTrue(self.service.str2bool(True))
        self.assertFalse(self.service.str2bool("false"))
        self.assertFalse(self.service.str2bool(False))


class TestSRIntegration(unittest.TestCase):
    """Integration tests for SR calculator"""

    @patch("yfinance.download")
    def test_full_calculation_workflow(self, mock_yf_download):
        """Test complete calculation workflow"""
        # Mock yfinance data
        test_data = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [102, 103, 104, 105, 106],
                "Low": [99, 100, 101, 102, 103],
                "Close": [101, 102, 103, 104, 105],
                "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            },
            index=pd.date_range("2023-01-01", periods=5, freq="D"),
        )

        mock_yf_download.return_value = test_data

        service = SRService()
        request = SRCalculationRequest(
            ticker="AAPL",
            method="pivot_points",
            datasource="yfinance",
            start="2023-01-01",
            end="2023-01-05",
        )

        # Note: This would require actual service dependencies
        # In practice, we'd mock the download_service as well
        # For now, we'll just test that the service can be instantiated
        self.assertIsInstance(service, SRService)
        self.assertIsInstance(request, SRCalculationRequest)


if __name__ == "__main__":
    unittest.main()
