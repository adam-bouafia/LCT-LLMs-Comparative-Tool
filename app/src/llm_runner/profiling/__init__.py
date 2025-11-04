"""
Energy Profiling Module

Provides comprehensive energy consumption tracking and analysis for LLM evaluation.
Includes environmental impact tracking (water, PUE, eco-efficiency).
"""

from .energy_tracker import (
    EnergyTracker,
    EnergyStage,
    EnergyMeasurement,
    AlgorithmEnergyProfile,
    PromptEnergyProfile,
)

from .environmental_impact import (
    EnvironmentalImpactCalculator,
    EnvironmentalImpact,
    EnvironmentalMultipliers,
    EcoEfficiencyScore,
    Region,
)

__all__ = [
    "EnergyTracker",
    "EnergyStage",
    "EnergyMeasurement",
    "AlgorithmEnergyProfile",
    "PromptEnergyProfile",
    "EnvironmentalImpactCalculator",
    "EnvironmentalImpact",
    "EnvironmentalMultipliers",
    "EcoEfficiencyScore",
    "Region",
]
