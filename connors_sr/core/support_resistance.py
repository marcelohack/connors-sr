"""
Core Support & Resistance interfaces and calculation methods

This module defines the core interfaces and calculation methods for
Support & Resistance level identification using various market analysis techniques.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd


class SRMethod(Enum):
    """Enumeration of available Support & Resistance calculation methods"""

    PIVOT_POINTS = "pivot_points"
    FRACTAL = "fractal"
    VWAP_ZONES = "vwap_zones"
    VOLUME_PROFILE = "volume_profile"


@dataclass
class SRLevel:
    """Container for a single Support/Resistance level"""

    level: float
    level_type: str  # 'support' or 'resistance'
    method: SRMethod
    strength: float  # Strength indicator (0.0 to 1.0)
    touches: int  # Number of times price touched this level
    first_occurrence: pd.Timestamp
    last_occurrence: pd.Timestamp
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SRResult:
    """Container for Support & Resistance calculation results"""

    ticker: str
    data: pd.DataFrame  # Original OHLCV data with SR columns added
    levels: List[SRLevel]  # Identified support/resistance levels
    method: SRMethod
    parameters: Dict[str, Any]
    calculation_time: float
    success: bool = True
    error: Optional[str] = None


class SRCalculator(Protocol):
    """Protocol for Support & Resistance calculators"""

    def calculate(self, data: pd.DataFrame, **params: Any) -> SRResult:
        """Calculate Support & Resistance levels from OHLCV data"""
        ...

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for this calculation method"""
        ...

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available parameters"""
        ...


class BaseSRCalculator(ABC):
    """Base class for Support & Resistance calculators"""

    def __init__(self, method: SRMethod):
        self.method = method

    @abstractmethod
    def calculate(self, data: pd.DataFrame, **params: Any) -> SRResult:
        """Calculate Support & Resistance levels"""
        pass

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters"""
        pass

    @abstractmethod
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information"""
        pass

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input OHLCV data"""
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if len(data) < 10:  # Minimum data points needed
            raise ValueError("Insufficient data points for calculation (minimum: 10)")

    def _add_sr_columns(
        self, data: pd.DataFrame, levels: List[SRLevel], method_suffix: str = ""
    ) -> pd.DataFrame:
        """Add support/resistance columns to the dataframe"""
        df = data.copy()

        # Create column names
        support_col = f"Support_{self.method.value.title()}{method_suffix}"
        resistance_col = f"Resistance_{self.method.value.title()}{method_suffix}"

        # Initialize columns
        df[support_col] = np.nan
        df[resistance_col] = np.nan

        # Add levels to appropriate columns
        for level in levels:
            # Find nearest price points to the level
            if level.level_type == "support":
                # Mark support level where low prices are close to the level
                mask = abs(df["Low"] - level.level) <= (
                    level.level * 0.02
                )  # 2% tolerance
                df.loc[mask, support_col] = level.level
            elif level.level_type == "resistance":
                # Mark resistance level where high prices are close to the level
                mask = abs(df["High"] - level.level) <= (
                    level.level * 0.02
                )  # 2% tolerance
                df.loc[mask, resistance_col] = level.level

        return df


class PivotPointsCalculator(BaseSRCalculator):
    """Traditional Pivot Points calculation for Support & Resistance"""

    def __init__(self):
        super().__init__(SRMethod.PIVOT_POINTS)

    def calculate(
        self,
        data: pd.DataFrame,
        period: int = 1,
        include_midpoints: bool = True,
        **params: Any,
    ) -> SRResult:
        """
        Calculate Pivot Points Support & Resistance levels

        Args:
            data: OHLCV DataFrame
            period: Number of periods to calculate pivots (1 = daily)
            include_midpoints: Whether to include midpoint levels
        """
        import time

        start_time = time.time()

        self._validate_data(data)

        try:
            levels = []
            df = data.copy()

            # Use a rolling window approach to find significant pivot points
            # that actually represent meaningful support and resistance levels

            window = max(period * 5, 20)  # Use a reasonable window size

            # Find local highs and lows (swing points)
            df["LocalHigh"] = df["High"][
                (df["High"].shift(period) < df["High"])
                & (df["High"] > df["High"].shift(-period))
            ]
            df["LocalLow"] = df["Low"][
                (df["Low"].shift(period) > df["Low"])
                & (df["Low"] < df["Low"].shift(-period))
            ]

            # Calculate traditional pivot points for reference
            df["PP"] = (df["High"] + df["Low"] + df["Close"]) / 3  # Current period PP
            df["R1"] = 2 * df["PP"] - df["Low"]
            df["S1"] = 2 * df["PP"] - df["High"]
            df["R2"] = df["PP"] + (df["High"] - df["Low"])
            df["S2"] = df["PP"] - (df["High"] - df["Low"])
            df["R3"] = df["High"] + 2 * (df["PP"] - df["Low"])
            df["S3"] = df["Low"] - 2 * (df["High"] - df["PP"])

            if include_midpoints:
                df["M1"] = (df["PP"] + df["S1"]) / 2
                df["M2"] = (df["PP"] + df["R1"]) / 2
                df["M3"] = (df["S1"] + df["S2"]) / 2
                df["M4"] = (df["R1"] + df["R2"]) / 2

            # Find significant support levels from local lows and calculated support levels
            price_min = df["Low"].min()
            price_max = df["High"].max()
            price_range = price_max - price_min

            # Collect potential support levels
            support_candidates = []

            # From local lows (actual price levels where support was found)
            local_lows = df["LocalLow"].dropna()
            for low_val in local_lows:
                if not np.isnan(low_val):
                    support_candidates.append(low_val)

            # From calculated pivot support levels that are within reasonable range
            for col in ["S1", "S2", "S3"] + (["M1", "M3"] if include_midpoints else []):
                pivot_supports = df[col].dropna()
                for support_val in pivot_supports:
                    if (
                        not np.isnan(support_val)
                        and price_min <= support_val <= price_max
                    ):
                        support_candidates.append(support_val)

            # Group similar support levels and create SR levels
            support_levels_dict = {}
            for candidate in support_candidates:
                # Group levels within 1% of each other
                grouped = False
                for existing_level in support_levels_dict.keys():
                    if abs(candidate - existing_level) / existing_level <= 0.01:
                        support_levels_dict[existing_level].append(candidate)
                        grouped = True
                        break
                if not grouped:
                    support_levels_dict[candidate] = [candidate]

            # Create support SRLevels from significant groups
            for base_level, group_levels in support_levels_dict.items():
                if len(group_levels) >= 1:  # At least 1 occurrence
                    avg_level = np.mean(group_levels)
                    touches = self._count_touches(df, avg_level)

                    if touches >= 1:  # Must have at least 1 touch
                        levels.append(
                            SRLevel(
                                level=float(avg_level),
                                level_type="support",
                                method=self.method,
                                strength=min(
                                    len(group_levels) * 0.5, 2.0
                                ),  # Cap at 2.0
                                touches=touches,
                                first_occurrence=df.index[0],
                                last_occurrence=df.index[-1],
                                metadata={
                                    "pivot_type": "calculated",
                                    "occurrences": len(group_levels),
                                },
                            )
                        )

            # Find significant resistance levels from local highs and calculated resistance levels
            resistance_candidates = []

            # From local highs (actual price levels where resistance was found)
            local_highs = df["LocalHigh"].dropna()
            for high_val in local_highs:
                if not np.isnan(high_val):
                    resistance_candidates.append(high_val)

            # From calculated pivot resistance levels that are within reasonable range
            for col in ["R1", "R2", "R3"] + (["M2", "M4"] if include_midpoints else []):
                pivot_resistances = df[col].dropna()
                for resistance_val in pivot_resistances:
                    if (
                        not np.isnan(resistance_val)
                        and price_min <= resistance_val <= price_max
                    ):
                        resistance_candidates.append(resistance_val)

            # Group similar resistance levels and create SR levels
            resistance_levels_dict = {}
            for candidate in resistance_candidates:
                # Group levels within 1% of each other
                grouped = False
                for existing_level in resistance_levels_dict.keys():
                    if abs(candidate - existing_level) / existing_level <= 0.01:
                        resistance_levels_dict[existing_level].append(candidate)
                        grouped = True
                        break
                if not grouped:
                    resistance_levels_dict[candidate] = [candidate]

            # Create resistance SRLevels from significant groups
            for base_level, group_levels in resistance_levels_dict.items():
                if len(group_levels) >= 1:  # At least 1 occurrence
                    avg_level = np.mean(group_levels)
                    touches = self._count_touches(df, avg_level)

                    if touches >= 1:  # Must have at least 1 touch
                        levels.append(
                            SRLevel(
                                level=float(avg_level),
                                level_type="resistance",
                                method=self.method,
                                strength=min(
                                    len(group_levels) * 0.5, 2.0
                                ),  # Cap at 2.0
                                touches=touches,
                                first_occurrence=df.index[0],
                                last_occurrence=df.index[-1],
                                metadata={
                                    "pivot_type": "calculated",
                                    "occurrences": len(group_levels),
                                },
                            )
                        )

            # Add SR columns to dataframe
            result_df = self._add_sr_columns(df, levels)

            calculation_time = time.time() - start_time

            return SRResult(
                ticker=getattr(data, "ticker", "UNKNOWN"),
                data=result_df,
                levels=levels,
                method=self.method,
                parameters={"period": period, "include_midpoints": include_midpoints},
                calculation_time=calculation_time,
            )

        except Exception as e:
            return SRResult(
                ticker=getattr(data, "ticker", "UNKNOWN"),
                data=data,
                levels=[],
                method=self.method,
                parameters={"period": period, "include_midpoints": include_midpoints},
                calculation_time=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for Pivot Points calculation"""
        return {"period": 1, "include_midpoints": False}

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information"""
        return {
            "period": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 10,
                "description": "Number of periods to calculate pivots (1 = daily)",
            },
            "include_midpoints": {
                "type": "bool",
                "default": False,
                "description": "Whether to include midpoint levels between main pivots",
            },
        }

    def _calculate_level_strength(
        self, data: pd.DataFrame, level: float, level_type: str
    ) -> float:
        """Calculate the strength of a support/resistance level (0.0 to 1.0)"""
        if level_type == "support":
            # Count how many times price approached but didn't break the level
            near_level = abs(data["Low"] - level) <= (level * 0.01)  # 1% tolerance
        else:
            near_level = abs(data["High"] - level) <= (level * 0.01)

        touches = near_level.sum()
        max_touches = min(len(data) // 10, 10)  # Normalize by data length
        return min(touches / max_touches, 1.0) if max_touches > 0 else 0.0

    def _count_touches(self, data: pd.DataFrame, level: float) -> int:
        """Count how many times price touched the level"""
        tolerance = level * 0.01  # 1% tolerance
        touches = 0
        touches += (abs(data["High"] - level) <= tolerance).sum()
        touches += (abs(data["Low"] - level) <= tolerance).sum()
        touches += (abs(data["Close"] - level) <= tolerance).sum()
        return int(touches)


class FractalCalculator(BaseSRCalculator):
    """Fractal-based Support & Resistance using recent highs and lows"""

    def __init__(self):
        super().__init__(SRMethod.FRACTAL)

    def calculate(
        self,
        data: pd.DataFrame,
        lookback: int = 5,
        min_strength: float = 0.3,
        **params: Any,
    ) -> SRResult:
        """
        Calculate Fractal Support & Resistance levels

        Args:
            data: OHLCV DataFrame
            lookback: Periods to look back for fractal identification
            min_strength: Minimum strength threshold for level inclusion
        """
        import time

        start_time = time.time()

        self._validate_data(data)

        try:
            levels = []
            df = data.copy()

            # Find fractal highs (resistance)
            df["FractalHigh"] = np.nan
            for i in range(lookback, len(df) - lookback):
                if (
                    df["High"].iloc[i]
                    == df["High"].iloc[i - lookback : i + lookback + 1].max()
                ):
                    df.loc[df.index[i], "FractalHigh"] = df["High"].iloc[i]

            # Find fractal lows (support)
            df["FractalLow"] = np.nan
            for i in range(lookback, len(df) - lookback):
                if (
                    df["Low"].iloc[i]
                    == df["Low"].iloc[i - lookback : i + lookback + 1].min()
                ):
                    df.loc[df.index[i], "FractalLow"] = df["Low"].iloc[i]

            # Process fractal highs as resistance levels
            fractal_highs = df["FractalHigh"].dropna()
            for timestamp, level_value in fractal_highs.items():
                strength = self._calculate_level_strength(df, level_value, "resistance")
                if strength >= min_strength:
                    levels.append(
                        SRLevel(
                            level=float(level_value),
                            level_type="resistance",
                            method=self.method,
                            strength=strength,
                            touches=self._count_touches(df, level_value),
                            first_occurrence=timestamp,
                            last_occurrence=timestamp,
                            metadata={"fractal_type": "high", "lookback": lookback},
                        )
                    )

            # Process fractal lows as support levels
            fractal_lows = df["FractalLow"].dropna()
            for timestamp, level_value in fractal_lows.items():
                strength = self._calculate_level_strength(df, level_value, "support")
                if strength >= min_strength:
                    levels.append(
                        SRLevel(
                            level=float(level_value),
                            level_type="support",
                            method=self.method,
                            strength=strength,
                            touches=self._count_touches(df, level_value),
                            first_occurrence=timestamp,
                            last_occurrence=timestamp,
                            metadata={"fractal_type": "low", "lookback": lookback},
                        )
                    )

            # Add SR columns to dataframe
            result_df = self._add_sr_columns(df, levels, f"_{lookback}")

            calculation_time = time.time() - start_time

            return SRResult(
                ticker=getattr(data, "ticker", "UNKNOWN"),
                data=result_df,
                levels=levels,
                method=self.method,
                parameters={"lookback": lookback, "min_strength": min_strength},
                calculation_time=calculation_time,
            )

        except Exception as e:
            return SRResult(
                ticker=getattr(data, "ticker", "UNKNOWN"),
                data=data,
                levels=[],
                method=self.method,
                parameters={"lookback": lookback, "min_strength": min_strength},
                calculation_time=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for Fractal calculation"""
        return {"lookback": 5, "min_strength": 0.3}

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information"""
        return {
            "lookback": {
                "type": "int",
                "default": 5,
                "min": 3,
                "max": 20,
                "description": "Periods to look back for fractal identification",
            },
            "min_strength": {
                "type": "float",
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum strength threshold for level inclusion (0.0 to 1.0)",
            },
        }

    def _calculate_level_strength(
        self, data: pd.DataFrame, level: float, level_type: str
    ) -> float:
        """Calculate the strength of a fractal level based on subsequent price action"""
        tolerance = level * 0.015  # 1.5% tolerance

        if level_type == "support":
            # Count rebounds from support level
            near_level = abs(data["Low"] - level) <= tolerance
            subsequent_bounces = 0
            for i in data[near_level].index:
                idx = data.index.get_loc(i)
                if idx < len(data) - 5:  # Look ahead 5 periods
                    future_low = data["Low"].iloc[idx : idx + 5].min()
                    if data["Low"].iloc[idx] <= future_low * 1.01:  # Price bounced up
                        subsequent_bounces += 1
        else:
            # Count rejections from resistance level
            near_level = abs(data["High"] - level) <= tolerance
            subsequent_bounces = 0
            for i in data[near_level].index:
                idx = data.index.get_loc(i)
                if idx < len(data) - 5:
                    future_high = data["High"].iloc[idx : idx + 5].max()
                    if (
                        data["High"].iloc[idx] >= future_high * 0.99
                    ):  # Price rejected down
                        subsequent_bounces += 1

        max_possible_bounces = min(near_level.sum(), 5)
        return (
            subsequent_bounces / max_possible_bounces
            if max_possible_bounces > 0
            else 0.0
        )

    def _count_touches(self, data: pd.DataFrame, level: float) -> int:
        """Count touches for fractal levels"""
        tolerance = level * 0.015  # 1.5% tolerance
        touches = 0
        touches += (abs(data["High"] - level) <= tolerance).sum()
        touches += (abs(data["Low"] - level) <= tolerance).sum()
        return int(touches)


class VWAPZonesCalculator(BaseSRCalculator):
    """VWAP-based Support & Resistance zones"""

    def __init__(self):
        super().__init__(SRMethod.VWAP_ZONES)

    def calculate(
        self,
        data: pd.DataFrame,
        period: int = 20,
        std_devs: List[float] = [1.0, 2.0],
        min_volume_ratio: float = 1.5,
        **params: Any,
    ) -> SRResult:
        """
        Calculate VWAP Zones Support & Resistance levels

        Args:
            data: OHLCV DataFrame
            period: Rolling period for VWAP calculation
            std_devs: Standard deviation multipliers for bands
            min_volume_ratio: Minimum volume ratio vs average to consider level significant
        """
        import time

        start_time = time.time()

        self._validate_data(data)

        try:
            levels = []
            df = data.copy()

            # Calculate typical price
            df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3

            # Calculate VWAP
            df["CumVolume"] = df["Volume"].rolling(period).sum()
            df["CumVolumePrice"] = (df["TP"] * df["Volume"]).rolling(period).sum()
            df["VWAP"] = df["CumVolumePrice"] / df["CumVolume"]

            # Calculate standard deviation of typical price around VWAP
            df["TP_VWAP_Diff"] = df["TP"] - df["VWAP"]
            df["VWAP_Std"] = df["TP_VWAP_Diff"].rolling(period).std()

            # Calculate VWAP bands
            for i, std_dev in enumerate(std_devs):
                df[f"VWAP_Upper_{i+1}"] = df["VWAP"] + (df["VWAP_Std"] * std_dev)
                df[f"VWAP_Lower_{i+1}"] = df["VWAP"] - (df["VWAP_Std"] * std_dev)

            # Calculate average volume for comparison
            df["AvgVolume"] = df["Volume"].rolling(period).mean()

            # Identify significant VWAP levels based on high volume
            high_volume_mask = df["Volume"] >= (df["AvgVolume"] * min_volume_ratio)
            significant_data = df[high_volume_mask].tail(
                50
            )  # Last 50 high-volume periods

            # Process VWAP as pivot level
            for timestamp, row in significant_data.iterrows():
                vwap_level = row["VWAP"]
                if not np.isnan(vwap_level):
                    # Determine if acting as support or resistance based on recent price action
                    recent_closes = df.loc[:timestamp, "Close"].tail(5)
                    if recent_closes.mean() > vwap_level:
                        level_type = "support"
                    else:
                        level_type = "resistance"

                    levels.append(
                        SRLevel(
                            level=float(vwap_level),
                            level_type=level_type,
                            method=self.method,
                            strength=self._calculate_vwap_strength(
                                df, vwap_level, row["Volume"], row["AvgVolume"]
                            ),
                            touches=self._count_vwap_touches(df, vwap_level),
                            first_occurrence=timestamp,
                            last_occurrence=timestamp,
                            metadata={
                                "level_type": "vwap",
                                "volume_ratio": row["Volume"] / row["AvgVolume"],
                            },
                        )
                    )

            # Process VWAP bands as support/resistance
            for i, std_dev in enumerate(std_devs):
                upper_col = f"VWAP_Upper_{i+1}"
                lower_col = f"VWAP_Lower_{i+1}"

                # Upper bands as resistance
                for timestamp, level_value in (
                    significant_data[upper_col].dropna().items()
                ):
                    levels.append(
                        SRLevel(
                            level=float(level_value),
                            level_type="resistance",
                            method=self.method,
                            strength=self._calculate_band_strength(
                                df, level_value, std_dev
                            ),
                            touches=self._count_vwap_touches(df, level_value),
                            first_occurrence=timestamp,
                            last_occurrence=timestamp,
                            metadata={"level_type": "upper_band", "std_dev": std_dev},
                        )
                    )

                # Lower bands as support
                for timestamp, level_value in (
                    significant_data[lower_col].dropna().items()
                ):
                    levels.append(
                        SRLevel(
                            level=float(level_value),
                            level_type="support",
                            method=self.method,
                            strength=self._calculate_band_strength(
                                df, level_value, std_dev
                            ),
                            touches=self._count_vwap_touches(df, level_value),
                            first_occurrence=timestamp,
                            last_occurrence=timestamp,
                            metadata={"level_type": "lower_band", "std_dev": std_dev},
                        )
                    )

            # Add SR columns to dataframe
            result_df = self._add_sr_columns(df, levels, f"_{period}")

            calculation_time = time.time() - start_time

            return SRResult(
                ticker=getattr(data, "ticker", "UNKNOWN"),
                data=result_df,
                levels=levels,
                method=self.method,
                parameters={
                    "period": period,
                    "std_devs": std_devs,
                    "min_volume_ratio": min_volume_ratio,
                },
                calculation_time=calculation_time,
            )

        except Exception as e:
            return SRResult(
                ticker=getattr(data, "ticker", "UNKNOWN"),
                data=data,
                levels=[],
                method=self.method,
                parameters={
                    "period": period,
                    "std_devs": std_devs,
                    "min_volume_ratio": min_volume_ratio,
                },
                calculation_time=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for VWAP Zones calculation"""
        return {"period": 20, "std_devs": [1.0, 2.0], "min_volume_ratio": 1.5}

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information"""
        return {
            "period": {
                "type": "int",
                "default": 20,
                "min": 10,
                "max": 100,
                "description": "Rolling period for VWAP calculation",
            },
            "std_devs": {
                "type": "list",
                "default": [1.0, 2.0],
                "description": "Standard deviation multipliers for VWAP bands",
            },
            "min_volume_ratio": {
                "type": "float",
                "default": 1.5,
                "min": 1.0,
                "max": 5.0,
                "description": "Minimum volume ratio vs average to consider level significant",
            },
        }

    def _calculate_vwap_strength(
        self, data: pd.DataFrame, level: float, volume: float, avg_volume: float
    ) -> float:
        """Calculate VWAP level strength based on volume and price action"""
        # Base strength on volume ratio
        volume_strength = min(volume / avg_volume / 3.0, 1.0)  # Normalize to max 1.0

        # Adjust by how well the level holds
        tolerance = level * 0.01
        near_touches = abs(data["Close"] - level) <= tolerance
        successful_bounces = 0

        for i in data[near_touches].index:
            idx = data.index.get_loc(i)
            if idx < len(data) - 3:
                future_prices = data["Close"].iloc[idx : idx + 3]
                if len(future_prices) > 1:
                    if abs(future_prices.iloc[0] - level) > abs(
                        future_prices.iloc[-1] - level
                    ):
                        successful_bounces += 1

        bounce_strength = successful_bounces / max(near_touches.sum(), 1)
        return (volume_strength + bounce_strength) / 2

    def _calculate_band_strength(
        self, data: pd.DataFrame, level: float, std_dev: float
    ) -> float:
        """Calculate VWAP band strength"""
        # Higher standard deviation bands are stronger
        base_strength = min(std_dev / 3.0, 1.0)  # 3 std dev = max strength

        # Count successful bounces off the band
        tolerance = level * 0.015
        touches = abs(data["High"] - level) <= tolerance
        touches |= abs(data["Low"] - level) <= tolerance

        bounce_strength = min(touches.sum() / 10.0, 1.0)  # Normalize
        return (base_strength + bounce_strength) / 2

    def _count_vwap_touches(self, data: pd.DataFrame, level: float) -> int:
        """Count touches for VWAP levels"""
        tolerance = level * 0.01
        touches = 0
        touches += (abs(data["High"] - level) <= tolerance).sum()
        touches += (abs(data["Low"] - level) <= tolerance).sum()
        touches += (abs(data["Close"] - level) <= tolerance).sum()
        return int(touches)


class VolumeProfileCalculator(BaseSRCalculator):
    """Volume Profile-based Support & Resistance levels"""

    def __init__(self):
        super().__init__(SRMethod.VOLUME_PROFILE)

    def calculate(
        self,
        data: pd.DataFrame,
        price_bins: int = 50,
        min_volume_pct: float = 5.0,
        lookback_periods: int = 100,
        **params: Any,
    ) -> SRResult:
        """
        Calculate Volume Profile Support & Resistance levels

        Args:
            data: OHLCV DataFrame
            price_bins: Number of price bins to create for volume distribution
            min_volume_pct: Minimum volume percentage to consider level significant
            lookback_periods: Number of periods to look back for volume profile
        """
        import time

        start_time = time.time()

        self._validate_data(data)

        try:
            levels = []
            df = data.copy()

            # Use recent data for volume profile
            recent_data = df.tail(lookback_periods).copy()

            # Create price bins
            price_range = recent_data["High"].max() - recent_data["Low"].min()
            bin_size = price_range / price_bins

            # Calculate volume at each price level
            price_volume_map = {}

            for _, row in recent_data.iterrows():
                # Distribute volume across the trading range for each period
                high, low, volume = row["High"], row["Low"], row["Volume"]
                price_range_period = high - low

                if price_range_period > 0:
                    # Number of price bins this period spans
                    bins_spanned = max(1, int(price_range_period / bin_size))
                    volume_per_bin = volume / bins_spanned

                    # Distribute volume across bins
                    for i in range(bins_spanned):
                        price_level = low + (i * (price_range_period / bins_spanned))
                        bin_key = round(price_level / bin_size) * bin_size
                        price_volume_map[bin_key] = (
                            price_volume_map.get(bin_key, 0) + volume_per_bin
                        )

            # Calculate total volume and find high volume nodes
            total_volume = sum(price_volume_map.values())
            high_volume_levels = []

            for price_level, volume in price_volume_map.items():
                volume_pct = (volume / total_volume) * 100
                if volume_pct >= min_volume_pct:
                    high_volume_levels.append((price_level, volume, volume_pct))

            # Sort by volume percentage (descending)
            high_volume_levels.sort(key=lambda x: x[2], reverse=True)

            # Create support/resistance levels from high volume nodes
            for price_level, volume, volume_pct in high_volume_levels:
                # Determine if acting as support or resistance based on current price position
                current_price = recent_data["Close"].iloc[-1]

                if price_level < current_price:
                    level_type = "support"
                else:
                    level_type = "resistance"

                levels.append(
                    SRLevel(
                        level=float(price_level),
                        level_type=level_type,
                        method=self.method,
                        strength=min(
                            volume_pct / 20.0, 1.0
                        ),  # Normalize strength (20% volume = max strength)
                        touches=self._count_volume_profile_touches(
                            recent_data, price_level, bin_size
                        ),
                        first_occurrence=recent_data.index[0],
                        last_occurrence=recent_data.index[-1],
                        metadata={
                            "volume_pct": volume_pct,
                            "total_volume": volume,
                            "price_bin_size": bin_size,
                            "lookback_periods": lookback_periods,
                        },
                    )
                )

            # Add volume profile columns to dataframe
            df["VP_Volume"] = 0.0
            df["VP_Level"] = np.nan

            # Mark significant volume levels in the dataframe
            tolerance = bin_size / 2
            for level_obj in levels[:10]:  # Top 10 levels only
                level_value = level_obj.level
                mask = (df["High"] >= level_value - tolerance) & (
                    df["Low"] <= level_value + tolerance
                )
                df.loc[mask, "VP_Level"] = level_value
                df.loc[mask, "VP_Volume"] = level_obj.metadata["volume_pct"]

            # Add SR columns to dataframe
            result_df = self._add_sr_columns(df, levels, f"_VP{price_bins}")

            calculation_time = time.time() - start_time

            return SRResult(
                ticker=getattr(data, "ticker", "UNKNOWN"),
                data=result_df,
                levels=levels,
                method=self.method,
                parameters={
                    "price_bins": price_bins,
                    "min_volume_pct": min_volume_pct,
                    "lookback_periods": lookback_periods,
                },
                calculation_time=calculation_time,
            )

        except Exception as e:
            return SRResult(
                ticker=getattr(data, "ticker", "UNKNOWN"),
                data=data,
                levels=[],
                method=self.method,
                parameters={
                    "price_bins": price_bins,
                    "min_volume_pct": min_volume_pct,
                    "lookback_periods": lookback_periods,
                },
                calculation_time=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for Volume Profile calculation"""
        return {"price_bins": 50, "min_volume_pct": 5.0, "lookback_periods": 100}

    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information"""
        return {
            "price_bins": {
                "type": "int",
                "default": 50,
                "min": 20,
                "max": 200,
                "description": "Number of price bins to create for volume distribution",
            },
            "min_volume_pct": {
                "type": "float",
                "default": 5.0,
                "min": 1.0,
                "max": 20.0,
                "description": "Minimum volume percentage to consider level significant",
            },
            "lookback_periods": {
                "type": "int",
                "default": 100,
                "min": 50,
                "max": 500,
                "description": "Number of periods to look back for volume profile analysis",
            },
        }

    def _count_volume_profile_touches(
        self, data: pd.DataFrame, level: float, bin_size: float
    ) -> int:
        """Count touches for volume profile levels"""
        tolerance = bin_size / 2
        touches = 0
        touches += (
            (data["High"] >= level - tolerance) & (data["High"] <= level + tolerance)
        ).sum()
        touches += (
            (data["Low"] >= level - tolerance) & (data["Low"] <= level + tolerance)
        ).sum()
        touches += (
            (data["Close"] >= level - tolerance) & (data["Close"] <= level + tolerance)
        ).sum()
        return int(touches)
