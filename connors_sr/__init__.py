"""
Connors SR - Support & Resistance Calculator

A standalone package for identifying Support & Resistance levels using
various technical analysis methods.
"""

from connors_sr.core.registry import registry
from connors_sr.core.support_resistance import (
    BaseSRCalculator,
    FractalCalculator,
    PivotPointsCalculator,
    SRCalculator,
    SRLevel,
    SRMethod,
    SRResult,
    VolumeProfileCalculator,
    VWAPZonesCalculator,
)
from connors_sr.services.sr_service import (
    SRCalculationRequest,
    SRService,
    SRServiceResult,
)
from connors_sr.version import __version__

__all__ = [
    # Core
    "SRMethod",
    "SRLevel",
    "SRResult",
    "SRCalculator",
    "BaseSRCalculator",
    "PivotPointsCalculator",
    "FractalCalculator",
    "VolumeProfileCalculator",
    "VWAPZonesCalculator",
    # Service
    "SRService",
    "SRCalculationRequest",
    "SRServiceResult",
    # Registry
    "registry",
    # Version
    "__version__",
]
