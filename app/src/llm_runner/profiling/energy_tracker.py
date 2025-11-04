"""
Energy Tracker Module

Provides granular energy consumption tracking for LLM evaluation pipeline stages.
Tracks energy usage per stage: dataset loading, model initialization, inference, and metric computation.

Enhanced with comprehensive environmental impact tracking:
- Water Usage Effectiveness (WUE) - on-site and off-site water consumption
- Power Usage Effectiveness (PUE) - data center infrastructure overhead
- Eco-Efficiency Scoring - performance per unit of environmental cost
"""

import time
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .environmental_impact import (
    EnvironmentalImpactCalculator,
    EnvironmentalImpact,
    EcoEfficiencyScore,
    Region,
    EnvironmentalMultipliers
)

logger = logging.getLogger(__name__)


class EnergyStage(Enum):
    """Pipeline stages for energy tracking."""

    DATASET_LOADING = "dataset_loading"
    MODEL_INITIALIZATION = "model_initialization"
    INFERENCE = "inference"
    METRIC_CALCULATION = "metric_calculation"
    TOTAL = "total"


@dataclass
class EnergyMeasurement:
    """Single energy measurement record."""

    stage: EnergyStage
    start_time: float
    end_time: Optional[float] = None
    energy_joules: float = 0.0
    power_watts: float = 0.0
    cpu_energy_kwh: float = 0.0
    gpu_energy_kwh: float = 0.0
    ram_energy_kwh: float = 0.0
    emissions_kg_co2: float = 0.0

    @property
    def duration_seconds(self) -> float:
        """Calculate duration in seconds."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, float]:
        """Convert measurement to dictionary."""
        return {
            "duration_seconds": self.duration_seconds,
            "energy_joules": self.energy_joules,
            "power_watts": self.power_watts,
            "cpu_energy_kwh": self.cpu_energy_kwh,
            "gpu_energy_kwh": self.gpu_energy_kwh,
            "ram_energy_kwh": self.ram_energy_kwh,
            "emissions_kg_co2": self.emissions_kg_co2,
        }


@dataclass
class AlgorithmEnergyProfile:
    """Energy consumption profile for a single algorithm."""

    algorithm_name: str
    start_time: float
    end_time: Optional[float] = None
    energy_joules: float = 0.0
    power_watts: float = 0.0
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "execution_time_ms": self.execution_time_ms,
            "energy_joules": self.energy_joules,
            "power_watts": self.power_watts,
        }


@dataclass
class PromptEnergyProfile:
    """Energy consumption profile for a single prompt evaluation."""

    prompt_index: int
    prompt_text: str
    start_time: float
    end_time: Optional[float] = None
    total_energy_joules: float = 0.0
    inference_energy_joules: float = 0.0
    metrics_energy_joules: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_energy_joules": self.total_energy_joules,
            "inference_energy_joules": self.inference_energy_joules,
            "metrics_energy_joules": self.metrics_energy_joules,
        }


class EnergyTracker:
    """
    Granular energy consumption tracker for LLM evaluation pipeline.

    Tracks energy consumption across multiple dimensions:
    - Pipeline stages (dataset loading, model init, inference, metrics)
    - Individual algorithms
    - Individual prompts
    - Total consumption
    """

    def __init__(
        self,
        profiler_type: str = "none",
        region: Region = Region.USA_EAST,
        enable_environmental_tracking: bool = True
    ):
        """
        Initialize energy tracker.

        Args:
            profiler_type: Type of energy profiler (codecarbon, none)
            region: Geographic region for environmental multipliers
            enable_environmental_tracking: Enable water, PUE, and eco-efficiency tracking
        """
        self.profiler_type = profiler_type
        self.stage_measurements: Dict[EnergyStage, List[EnergyMeasurement]] = {
            stage: [] for stage in EnergyStage
        }
        self.algorithm_profiles: Dict[str, AlgorithmEnergyProfile] = {}
        self.prompt_profiles: Dict[int, PromptEnergyProfile] = {}
        self.current_stage: Optional[EnergyStage] = None
        self.current_measurement: Optional[EnergyMeasurement] = None
        self.emissions_tracker = None
        
        # Environmental impact tracking
        self.enable_environmental_tracking = enable_environmental_tracking
        self.environmental_calculator: Optional[EnvironmentalImpactCalculator] = None
        
        if enable_environmental_tracking:
            self.environmental_calculator = EnvironmentalImpactCalculator(region=region)
            logger.info(f"Environmental tracking enabled for region: {region.value}")

        logger.info(f"Energy tracker initialized with profiler: {profiler_type}")

    def start_stage(self, stage: EnergyStage) -> None:
        """
        Start tracking energy for a pipeline stage.

        Args:
            stage: Pipeline stage to track
        """
        if self.current_stage is not None:
            logger.warning(
                f"Stage {self.current_stage} still active, stopping it first"
            )
            self.stop_stage()

        self.current_stage = stage
        self.current_measurement = EnergyMeasurement(
            stage=stage, start_time=time.time()
        )

        if self.profiler_type == "codecarbon" and stage != EnergyStage.TOTAL:
            self._start_codecarbon_stage()

        logger.debug(f"Started tracking stage: {stage.value}")

    def stop_stage(self) -> Optional[EnergyMeasurement]:
        """
        Stop tracking current stage and record measurement.

        Returns:
            Energy measurement for the completed stage
        """
        if self.current_stage is None or self.current_measurement is None:
            logger.warning("No active stage to stop")
            return None

        self.current_measurement.end_time = time.time()

        if self.profiler_type == "codecarbon":
            self._stop_codecarbon_stage()

        self.stage_measurements[self.current_stage].append(self.current_measurement)

        logger.debug(
            f"Stopped tracking stage: {self.current_stage.value} "
            f"(duration: {self.current_measurement.duration_seconds:.2f}s)"
        )

        completed_measurement = self.current_measurement
        self.current_stage = None
        self.current_measurement = None

        return completed_measurement

    def start_algorithm(self, algorithm_name: str) -> None:
        """
        Start tracking energy for an algorithm.

        Args:
            algorithm_name: Name of the algorithm being executed
        """
        self.algorithm_profiles[algorithm_name] = AlgorithmEnergyProfile(
            algorithm_name=algorithm_name, start_time=time.time()
        )
        logger.debug(f"Started tracking algorithm: {algorithm_name}")

    def stop_algorithm(
        self, algorithm_name: str, energy_data: Optional[Dict] = None
    ) -> None:
        """
        Stop tracking energy for an algorithm.

        Args:
            algorithm_name: Name of the algorithm
            energy_data: Optional energy data from profiler
        """
        if algorithm_name not in self.algorithm_profiles:
            logger.warning(f"Algorithm {algorithm_name} not being tracked")
            return

        profile = self.algorithm_profiles[algorithm_name]
        profile.end_time = time.time()
        profile.execution_time_ms = (profile.end_time - profile.start_time) * 1000

        if energy_data:
            profile.energy_joules = energy_data.get("energy_joules", 0.0)
            profile.power_watts = energy_data.get("power_watts", 0.0)

        logger.debug(
            f"Stopped tracking algorithm: {algorithm_name} "
            f"(execution time: {profile.execution_time_ms:.2f}ms)"
        )

    def start_prompt(self, prompt_index: int, prompt_text: str) -> None:
        """
        Start tracking energy for a prompt evaluation.

        Args:
            prompt_index: Index of the prompt
            prompt_text: Text of the prompt
        """
        self.prompt_profiles[prompt_index] = PromptEnergyProfile(
            prompt_index=prompt_index, prompt_text=prompt_text, start_time=time.time()
        )
        logger.debug(f"Started tracking prompt {prompt_index}")

    def stop_prompt(
        self, prompt_index: int, energy_breakdown: Optional[Dict] = None
    ) -> None:
        """
        Stop tracking energy for a prompt.

        Args:
            prompt_index: Index of the prompt
            energy_breakdown: Optional breakdown of energy consumption
        """
        if prompt_index not in self.prompt_profiles:
            logger.warning(f"Prompt {prompt_index} not being tracked")
            return

        profile = self.prompt_profiles[prompt_index]
        profile.end_time = time.time()

        if energy_breakdown:
            profile.total_energy_joules = energy_breakdown.get("total", 0.0)
            profile.inference_energy_joules = energy_breakdown.get("inference", 0.0)
            profile.metrics_energy_joules = energy_breakdown.get("metrics", 0.0)

        logger.debug(f"Stopped tracking prompt {prompt_index}")

    def get_stage_summary(self, stage: EnergyStage) -> Dict[str, float]:
        """
        Get energy summary for a specific stage.

        Args:
            stage: Pipeline stage

        Returns:
            Dictionary with aggregated metrics
        """
        measurements = self.stage_measurements.get(stage, [])
        if not measurements:
            return {
                "count": 0,
                "total_energy_joules": 0.0,
                "avg_energy_joules": 0.0,
                "total_duration_seconds": 0.0,
                "avg_power_watts": 0.0,
            }

        total_energy = sum(m.energy_joules for m in measurements)
        total_duration = sum(m.duration_seconds for m in measurements)
        avg_power = total_energy / total_duration if total_duration > 0 else 0.0

        return {
            "count": len(measurements),
            "total_energy_joules": total_energy,
            "avg_energy_joules": total_energy / len(measurements),
            "total_duration_seconds": total_duration,
            "avg_power_watts": avg_power,
        }

    def get_algorithm_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get energy summary for all algorithms.

        Returns:
            Dictionary mapping algorithm names to their energy profiles
        """
        return {
            name: profile.to_dict() for name, profile in self.algorithm_profiles.items()
        }

    def get_prompt_summary(self) -> Dict[int, Dict[str, float]]:
        """
        Get energy summary for all prompts.

        Returns:
            Dictionary mapping prompt indices to their energy profiles
        """
        return {idx: profile.to_dict() for idx, profile in self.prompt_profiles.items()}

    def get_full_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive energy report.

        Returns:
            Complete energy consumption report with environmental impact
        """
        total_energy = sum(
            sum(m.energy_joules for m in measurements)
            for measurements in self.stage_measurements.values()
        )
        total_duration = sum(
            sum(m.duration_seconds for m in measurements)
            for measurements in self.stage_measurements.values()
        )
        avg_power = total_energy / total_duration if total_duration > 0 else 0.0

        report = {
            "profiler_type": self.profiler_type,
            "stages": {
                stage.value: self.get_stage_summary(stage) for stage in EnergyStage
            },
            "algorithms": self.get_algorithm_summary(),
            "prompts": self.get_prompt_summary(),
            "total_energy_joules": total_energy,
            "total_duration_seconds": total_duration,
            "avg_power_watts": avg_power,
        }
        
        # Add environmental impact if tracking is enabled
        if self.enable_environmental_tracking and self.environmental_calculator:
            env_impact = self.get_environmental_impact()
            if env_impact:
                report["environmental_impact"] = env_impact.to_dict()
                report["environmental_impact_human_readable"] = env_impact.get_human_readable_summary()
        
        return report
    
    def get_environmental_impact(self) -> Optional[EnvironmentalImpact]:
        """
        Calculate comprehensive environmental impact for total energy consumption.
        
        Returns:
            EnvironmentalImpact with water, PUE-adjusted energy, and carbon metrics
        """
        if not self.enable_environmental_tracking or not self.environmental_calculator:
            return None
        
        # Calculate total energy in kWh
        total_energy_joules = sum(
            sum(m.energy_joules for m in measurements)
            for measurements in self.stage_measurements.values()
        )
        energy_kwh = total_energy_joules / 3_600_000  # Convert J to kWh
        
        # Get total carbon emissions from measurements
        total_carbon_kg = sum(
            sum(m.emissions_kg_co2 for m in measurements)
            for measurements in self.stage_measurements.values()
        )
        
        # Calculate comprehensive environmental impact
        impact = self.environmental_calculator.calculate_impact(
            energy_kwh=energy_kwh,
            carbon_kg=total_carbon_kg if total_carbon_kg > 0 else None
        )
        
        return impact
    
    def get_eco_efficiency_score(
        self,
        accuracy_score: Optional[float] = None,
        throughput_tokens_per_sec: Optional[float] = None,
        latency_seconds: Optional[float] = None
    ) -> Optional[EcoEfficiencyScore]:
        """
        Calculate eco-efficiency score balancing performance vs environmental cost.
        
        Args:
            accuracy_score: Model accuracy (0-1 or 0-100)
            throughput_tokens_per_sec: Tokens generated per second
            latency_seconds: Average latency in seconds
        
        Returns:
            EcoEfficiencyScore with calculated efficiency metric
        """
        if not self.enable_environmental_tracking or not self.environmental_calculator:
            return None
        
        impact = self.get_environmental_impact()
        if not impact:
            return None
        
        eco_score = self.environmental_calculator.calculate_eco_efficiency(
            impact=impact,
            accuracy_score=accuracy_score,
            throughput_tokens_per_sec=throughput_tokens_per_sec,
            latency_seconds=latency_seconds
        )
        
        return eco_score
    
    def get_scaled_impact_projection(
        self,
        queries_per_day: int,
        days: int = 365
    ) -> Optional[Dict[str, Any]]:
        """
        Project environmental impact at scale (e.g., production deployment).
        
        Args:
            queries_per_day: Number of queries per day
            days: Number of days to project (default: 365 for annual)
        
        Returns:
            Dictionary with scaled impact and human-readable equivalents
        """
        if not self.enable_environmental_tracking or not self.environmental_calculator:
            return None
        
        per_query_impact = self.get_environmental_impact()
        if not per_query_impact:
            return None
        
        scaled_impact, equivalents = self.environmental_calculator.calculate_scaled_impact(
            per_query_impact=per_query_impact,
            queries_per_day=queries_per_day,
            days=days
        )
        
        return {
            "scaled_impact": scaled_impact.to_dict(),
            "equivalents": equivalents
        }
    
    def set_region(self, region: Region) -> None:
        """
        Update environmental tracking region and multipliers.
        
        Args:
            region: New geographic region
        """
        if self.environmental_calculator:
            self.environmental_calculator.set_region(region)
            logger.info(f"Updated environmental tracking region to: {region.value}")
    
    def set_custom_multipliers(self, multipliers: EnvironmentalMultipliers) -> None:
        """
        Set custom environmental multipliers.
        
        Args:
            multipliers: Custom EnvironmentalMultipliers instance
        """
        if self.environmental_calculator:
            self.environmental_calculator.set_multipliers(multipliers)
            logger.info("Updated environmental multipliers with custom values")

    def _start_codecarbon_stage(self) -> None:
        """Start CodeCarbon tracking for current stage."""
        if self.current_stage is None:
            logger.warning("Cannot start CodeCarbon: no current stage set")
            return

        try:
            from codecarbon import EmissionsTracker

            self.emissions_tracker = EmissionsTracker(
                project_name=f"stage_{self.current_stage.value}",
                measure_power_secs=1,
                save_to_file=False,
                logging_logger=logger,  # type: ignore[arg-type]
            )
            self.emissions_tracker.start()
        except Exception as e:
            logger.error(f"Failed to start CodeCarbon for stage: {e}")
            self.emissions_tracker = None

    def _stop_codecarbon_stage(self) -> None:
        """Stop CodeCarbon tracking and record metrics."""
        if self.emissions_tracker is None or self.current_measurement is None:
            return

        try:
            emissions_data = self.emissions_tracker.stop()

            if emissions_data:
                self.current_measurement.energy_joules = emissions_data * 3600 * 1000
                self.current_measurement.cpu_energy_kwh = getattr(
                    self.emissions_tracker, "total_cpu_energy", 0
                )
                self.current_measurement.gpu_energy_kwh = getattr(
                    self.emissions_tracker, "total_gpu_energy", 0
                )
                self.current_measurement.ram_energy_kwh = getattr(
                    self.emissions_tracker, "total_ram_energy", 0
                )
                self.current_measurement.emissions_kg_co2 = emissions_data

                duration = self.current_measurement.duration_seconds
                if duration > 0:
                    self.current_measurement.power_watts = (
                        self.current_measurement.energy_joules / duration
                    )
        except Exception as e:
            logger.error(f"Failed to stop CodeCarbon for stage: {e}")

    def reset(self) -> None:
        """Reset all tracking data."""
        self.stage_measurements = {stage: [] for stage in EnergyStage}
        self.algorithm_profiles = {}
        self.prompt_profiles = {}
        self.current_stage = None
        self.current_measurement = None
        logger.info("Energy tracker reset")
