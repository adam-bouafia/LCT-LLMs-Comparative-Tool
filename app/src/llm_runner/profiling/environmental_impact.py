"""
Environmental Impact Calculator Module

Extends energy tracking with comprehensive environmental metrics:
- Water Usage Effectiveness (WUE) - on-site and off-site water consumption
- Power Usage Effectiveness (PUE) - data center infrastructure overhead
- Carbon Intensity Factors (CIF) - regional carbon emissions
- Eco-Efficiency Scoring - performance per unit of environmental cost
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Region(Enum):
    """Supported geographic regions with environmental multipliers."""
    # North America
    USA_EAST = "USA East"
    USA_WEST = "USA West"
    USA_CENTRAL = "USA Central"
    CANADA_EAST = "Canada East"
    CANADA_WEST = "Canada West"
    MEXICO = "Mexico"
    
    # Europe
    EUROPE_WEST = "Europe West"
    EUROPE_NORTH = "Europe North"
    EUROPE_EAST = "Europe East"
    EUROPE_SOUTH = "Europe South"
    UK = "United Kingdom"
    IRELAND = "Ireland"
    FRANCE = "France"
    GERMANY = "Germany"
    NETHERLANDS = "Netherlands"
    SWITZERLAND = "Switzerland"
    NORDICS = "Nordic Countries"
    
    # Asia Pacific
    CHINA_NORTH = "China North"
    CHINA_EAST = "China East"
    CHINA_SOUTH = "China South"
    JAPAN = "Japan"
    KOREA = "South Korea"
    SINGAPORE = "Singapore"
    AUSTRALIA_EAST = "Australia East"
    AUSTRALIA_SOUTHEAST = "Australia Southeast"
    INDIA_WEST = "India West"
    INDIA_SOUTH = "India South"
    INDIA_CENTRAL = "India Central"
    
    # Middle East & Africa
    UAE = "United Arab Emirates"
    SAUDI_ARABIA = "Saudi Arabia"
    SOUTH_AFRICA = "South Africa"
    
    # South America
    BRAZIL_SOUTH = "Brazil South"
    BRAZIL_SOUTHEAST = "Brazil Southeast"
    
    # Special Regions
    ICELAND = "Iceland"
    CUSTOM = "Custom"


@dataclass
class EnvironmentalMultipliers:
    """
    Environmental multipliers for a specific region.
    
    Attributes:
        pue: Power Usage Effectiveness (total DC energy / IT energy)
        wue_site: Water Usage Effectiveness for on-site cooling (L/kWh)
        wue_source: Water Usage Effectiveness for electricity generation (L/kWh)
        cif: Carbon Intensity Factor (kgCO2e/kWh)
    """
    region: Region
    pue: float = 1.20  # Default: 20% infrastructure overhead
    wue_site: float = 0.30  # Default: US average on-site water
    wue_source: float = 3.142  # Default: US average source water
    cif: float = 0.3528  # Default: US average carbon intensity
    
    @classmethod
    def get_regional_defaults(cls, region: Region) -> "EnvironmentalMultipliers":
        """
        Get default environmental multipliers for a region.
        
        Data sources: Cloud provider reports, EPA eGRID, IEA Emissions Factors, 
        regional energy mix data, and data center efficiency standards.
        """
        multipliers = {
            # North America - USA
            Region.USA_EAST: cls(
                region=Region.USA_EAST,
                pue=1.20,
                wue_site=0.30,
                wue_source=3.142,
                cif=0.3820  # Coal-heavy grid
            ),
            Region.USA_WEST: cls(
                region=Region.USA_WEST,
                pue=1.18,
                wue_site=0.25,
                wue_source=2.950,
                cif=0.2890  # More renewables
            ),
            Region.USA_CENTRAL: cls(
                region=Region.USA_CENTRAL,
                pue=1.22,
                wue_site=0.35,
                wue_source=3.280,
                cif=0.4150  # Coal and gas mix
            ),
            
            # North America - Canada
            Region.CANADA_EAST: cls(
                region=Region.CANADA_EAST,
                pue=1.14,
                wue_site=0.15,
                wue_source=1.850,
                cif=0.0800  # Hydroelectric heavy
            ),
            Region.CANADA_WEST: cls(
                region=Region.CANADA_WEST,
                pue=1.12,
                wue_site=0.12,
                wue_source=1.650,
                cif=0.0650  # Clean hydroelectric
            ),
            Region.MEXICO: cls(
                region=Region.MEXICO,
                pue=1.25,
                wue_site=0.45,
                wue_source=3.850,
                cif=0.4500
            ),
            
            # Europe - West
            Region.EUROPE_WEST: cls(
                region=Region.EUROPE_WEST,
                pue=1.15,
                wue_site=0.18,
                wue_source=2.800,
                cif=0.2950
            ),
            Region.UK: cls(
                region=Region.UK,
                pue=1.16,
                wue_site=0.20,
                wue_source=2.650,
                cif=0.2330  # Wind and gas
            ),
            Region.IRELAND: cls(
                region=Region.IRELAND,
                pue=1.12,
                wue_site=0.15,
                wue_source=2.200,
                cif=0.2580  # Wind heavy
            ),
            Region.FRANCE: cls(
                region=Region.FRANCE,
                pue=1.14,
                wue_site=0.16,
                wue_source=2.100,
                cif=0.0550  # Nuclear heavy
            ),
            Region.GERMANY: cls(
                region=Region.GERMANY,
                pue=1.17,
                wue_site=0.22,
                wue_source=2.950,
                cif=0.3380  # Coal transition
            ),
            Region.NETHERLANDS: cls(
                region=Region.NETHERLANDS,
                pue=1.13,
                wue_site=0.17,
                wue_source=2.450,
                cif=0.3050
            ),
            Region.SWITZERLAND: cls(
                region=Region.SWITZERLAND,
                pue=1.11,
                wue_site=0.12,
                wue_source=1.850,
                cif=0.0280  # Hydroelectric
            ),
            
            # Europe - North
            Region.EUROPE_NORTH: cls(
                region=Region.EUROPE_NORTH,
                pue=1.10,
                wue_site=0.10,
                wue_source=1.750,
                cif=0.0450
            ),
            Region.NORDICS: cls(
                region=Region.NORDICS,
                pue=1.09,
                wue_site=0.08,
                wue_source=1.550,
                cif=0.0350  # Renewable heavy
            ),
            Region.ICELAND: cls(
                region=Region.ICELAND,
                pue=1.08,
                wue_site=0.05,
                wue_source=0.500,
                cif=0.0100  # Geothermal
            ),
            
            # Europe - East & South
            Region.EUROPE_EAST: cls(
                region=Region.EUROPE_EAST,
                pue=1.24,
                wue_site=0.35,
                wue_source=3.850,
                cif=0.5200  # Coal heavy
            ),
            Region.EUROPE_SOUTH: cls(
                region=Region.EUROPE_SOUTH,
                pue=1.18,
                wue_site=0.28,
                wue_source=3.150,
                cif=0.3450
            ),
            
            # Asia Pacific - China
            Region.CHINA_NORTH: cls(
                region=Region.CHINA_NORTH,
                pue=1.28,
                wue_site=1.25,
                wue_source=6.200,
                cif=0.6500  # Coal heavy
            ),
            Region.CHINA_EAST: cls(
                region=Region.CHINA_EAST,
                pue=1.26,
                wue_site=1.15,
                wue_source=5.850,
                cif=0.5850
            ),
            Region.CHINA_SOUTH: cls(
                region=Region.CHINA_SOUTH,
                pue=1.25,
                wue_site=1.10,
                wue_source=5.650,
                cif=0.5550
            ),
            
            # Asia Pacific - Other
            Region.JAPAN: cls(
                region=Region.JAPAN,
                pue=1.19,
                wue_site=0.28,
                wue_source=3.450,
                cif=0.4680  # Post-nuclear transition
            ),
            Region.KOREA: cls(
                region=Region.KOREA,
                pue=1.21,
                wue_site=0.32,
                wue_source=3.850,
                cif=0.4320
            ),
            Region.SINGAPORE: cls(
                region=Region.SINGAPORE,
                pue=1.22,
                wue_site=0.35,
                wue_source=3.650,
                cif=0.3920  # Natural gas
            ),
            Region.AUSTRALIA_EAST: cls(
                region=Region.AUSTRALIA_EAST,
                pue=1.20,
                wue_site=0.38,
                wue_source=4.150,
                cif=0.7200  # Coal heavy
            ),
            Region.AUSTRALIA_SOUTHEAST: cls(
                region=Region.AUSTRALIA_SOUTHEAST,
                pue=1.18,
                wue_site=0.35,
                wue_source=3.950,
                cif=0.6850
            ),
            
            # India
            Region.INDIA_WEST: cls(
                region=Region.INDIA_WEST,
                pue=1.26,
                wue_site=0.85,
                wue_source=5.250,
                cif=0.7050  # Coal heavy
            ),
            Region.INDIA_SOUTH: cls(
                region=Region.INDIA_SOUTH,
                pue=1.24,
                wue_site=0.75,
                wue_source=4.950,
                cif=0.6750
            ),
            Region.INDIA_CENTRAL: cls(
                region=Region.INDIA_CENTRAL,
                pue=1.27,
                wue_site=0.90,
                wue_source=5.450,
                cif=0.7350
            ),
            
            # Middle East & Africa
            Region.UAE: cls(
                region=Region.UAE,
                pue=1.23,
                wue_site=0.55,
                wue_source=4.550,
                cif=0.4150  # Natural gas
            ),
            Region.SAUDI_ARABIA: cls(
                region=Region.SAUDI_ARABIA,
                pue=1.25,
                wue_site=0.65,
                wue_source=4.850,
                cif=0.4550
            ),
            Region.SOUTH_AFRICA: cls(
                region=Region.SOUTH_AFRICA,
                pue=1.24,
                wue_site=0.48,
                wue_source=4.350,
                cif=0.9500  # Coal dominant
            ),
            
            # South America
            Region.BRAZIL_SOUTH: cls(
                region=Region.BRAZIL_SOUTH,
                pue=1.19,
                wue_site=0.22,
                wue_source=2.650,
                cif=0.0850  # Hydroelectric heavy
            ),
            Region.BRAZIL_SOUTHEAST: cls(
                region=Region.BRAZIL_SOUTHEAST,
                pue=1.17,
                wue_site=0.20,
                wue_source=2.450,
                cif=0.0750
            ),
        }
        return multipliers.get(region, multipliers[Region.USA_EAST])


@dataclass
class EnvironmentalImpact:
    """
    Comprehensive environmental impact metrics.
    
    Attributes:
        energy_kwh_raw: Raw energy consumption (kWh)
        energy_kwh_with_pue: Energy with infrastructure overhead (kWh)
        water_liters_site: On-site water for cooling (liters)
        water_liters_source: Off-site water for electricity generation (liters)
        water_liters_total: Total water footprint (liters)
        carbon_kg: Carbon emissions (kg CO2e)
        pue_overhead_kwh: Extra energy from infrastructure (kWh)
    """
    energy_kwh_raw: float
    energy_kwh_with_pue: float
    water_liters_site: float
    water_liters_source: float
    water_liters_total: float
    carbon_kg: float
    pue_overhead_kwh: float
    region: Region
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "energy_kwh_raw": self.energy_kwh_raw,
            "energy_kwh_with_pue": self.energy_kwh_with_pue,
            "energy_kwh_infrastructure_overhead": self.pue_overhead_kwh,
            "water_liters_site_cooling": self.water_liters_site,
            "water_liters_source_generation": self.water_liters_source,
            "water_liters_total": self.water_liters_total,
            "carbon_kg_co2e": self.carbon_kg,
            "region": self.region.value,
        }
    
    def get_human_readable_summary(self) -> Dict[str, str]:
        """Generate human-readable environmental impact summary."""
        # Convert to more intuitive units
        energy_wh = self.energy_kwh_with_pue * 1000
        water_gallons = self.water_liters_total * 0.264172  # liters to gallons
        
        return {
            "Energy Consumption": f"{energy_wh:.2f} Wh ({self.energy_kwh_with_pue:.4f} kWh)",
            "Infrastructure Overhead": f"{self.pue_overhead_kwh * 1000:.2f} Wh",
            "Water Footprint": f"{self.water_liters_total:.2f} L ({water_gallons:.2f} gal)",
            "  - Cooling (on-site)": f"{self.water_liters_site:.2f} L",
            "  - Generation (off-site)": f"{self.water_liters_source:.2f} L",
            "Carbon Emissions": f"{self.carbon_kg:.4f} kg CO2e ({self.carbon_kg * 1000:.2f} g CO2e)",
            "Region": self.region.value,
        }


@dataclass
class EcoEfficiencyScore:
    """
    Eco-efficiency score balancing performance vs environmental cost.
    
    Based on Data Envelopment Analysis (DEA) methodology from paper.
    Higher score = better efficiency (more performance per unit of resource).
    """
    accuracy_score: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    latency_inverse: Optional[float] = None  # 1/latency for "higher is better"
    
    energy_kwh: float = 0.0
    water_liters: float = 0.0
    carbon_kg: float = 0.0
    
    efficiency_score: Optional[float] = None
    
    def calculate_efficiency(self) -> float:
        """
        Calculate eco-efficiency score.
        
        Formula: (Performance Outputs) / (Environmental Inputs)
        
        Returns:
            Efficiency score (higher is better)
        """
        # Aggregate performance metrics (outputs)
        performance = 0.0
        performance_count = 0
        
        if self.accuracy_score is not None:
            performance += self.accuracy_score
            performance_count += 1
        
        if self.throughput_tokens_per_sec is not None:
            # Normalize throughput (typical range 10-200 tokens/sec)
            normalized_throughput = min(self.throughput_tokens_per_sec / 100.0, 1.0)
            performance += normalized_throughput
            performance_count += 1
        
        if self.latency_inverse is not None:
            performance += self.latency_inverse
            performance_count += 1
        
        if performance_count == 0:
            logger.warning("No performance metrics available for efficiency calculation")
            return 0.0
        
        avg_performance = performance / performance_count
        
        # Aggregate environmental costs (inputs) - normalize to prevent division by zero
        env_cost = (
            max(self.energy_kwh, 0.001) +
            max(self.water_liters / 10.0, 0.001) +  # Scale down water
            max(self.carbon_kg * 10.0, 0.001)  # Scale up carbon
        )
        
        # Calculate efficiency: maximize performance, minimize environmental cost
        self.efficiency_score = avg_performance / env_cost
        
        return self.efficiency_score
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "accuracy_score": self.accuracy_score or 0.0,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec or 0.0,
            "latency_inverse": self.latency_inverse or 0.0,
            "energy_kwh": self.energy_kwh,
            "water_liters": self.water_liters,
            "carbon_kg": self.carbon_kg,
            "efficiency_score": self.efficiency_score or 0.0,
        }


class EnvironmentalImpactCalculator:
    """
    Calculator for comprehensive environmental impact assessment.
    
    Integrates:
    - Energy measurements (from CodeCarbon or other profilers)
    - Regional environmental multipliers (PUE, WUE, CIF)
    - Performance metrics (accuracy, throughput, latency)
    - Eco-efficiency scoring
    """
    
    def __init__(
        self,
        region: Region = Region.USA_EAST,
        custom_multipliers: Optional[EnvironmentalMultipliers] = None
    ):
        """
        Initialize environmental impact calculator.
        
        Args:
            region: Geographic region for environmental multipliers
            custom_multipliers: Optional custom multipliers (overrides region defaults)
        """
        self.region = region
        
        if custom_multipliers is not None:
            self.multipliers = custom_multipliers
        else:
            self.multipliers = EnvironmentalMultipliers.get_regional_defaults(region)
        
        logger.info(f"Environmental Impact Calculator initialized for region: {region.value}")
        logger.info(f"  PUE: {self.multipliers.pue:.2f}")
        logger.info(f"  WUE (site): {self.multipliers.wue_site:.2f} L/kWh")
        logger.info(f"  WUE (source): {self.multipliers.wue_source:.2f} L/kWh")
        logger.info(f"  CIF: {self.multipliers.cif:.4f} kgCO2e/kWh")
    
    def calculate_impact(
        self,
        energy_kwh: float,
        carbon_kg: Optional[float] = None
    ) -> EnvironmentalImpact:
        """
        Calculate comprehensive environmental impact from energy consumption.
        
        Args:
            energy_kwh: Raw energy consumption in kWh (IT equipment only)
            carbon_kg: Optional carbon emissions (if None, calculated from CIF)
        
        Returns:
            EnvironmentalImpact with all metrics calculated
        """
        # Apply PUE for realistic data center energy
        energy_with_pue = energy_kwh * self.multipliers.pue
        pue_overhead = energy_with_pue - energy_kwh
        
        # Calculate water consumption
        water_site = energy_with_pue * self.multipliers.wue_site
        water_source = energy_with_pue * self.multipliers.wue_source
        water_total = water_site + water_source
        
        # Calculate or use provided carbon emissions
        if carbon_kg is None:
            carbon_kg = energy_with_pue * self.multipliers.cif
        
        impact = EnvironmentalImpact(
            energy_kwh_raw=energy_kwh,
            energy_kwh_with_pue=energy_with_pue,
            water_liters_site=water_site,
            water_liters_source=water_source,
            water_liters_total=water_total,
            carbon_kg=carbon_kg,
            pue_overhead_kwh=pue_overhead,
            region=self.region
        )
        
        logger.debug(f"Calculated environmental impact: {energy_kwh:.4f} kWh -> "
                    f"{water_total:.2f} L water, {carbon_kg:.4f} kg CO2e")
        
        return impact
    
    def calculate_eco_efficiency(
        self,
        impact: EnvironmentalImpact,
        accuracy_score: Optional[float] = None,
        throughput_tokens_per_sec: Optional[float] = None,
        latency_seconds: Optional[float] = None
    ) -> EcoEfficiencyScore:
        """
        Calculate eco-efficiency score for model performance.
        
        Args:
            impact: Environmental impact metrics
            accuracy_score: Model accuracy (0-1 or 0-100)
            throughput_tokens_per_sec: Tokens generated per second
            latency_seconds: Response latency in seconds
        
        Returns:
            EcoEfficiencyScore with calculated efficiency
        """
        # Normalize accuracy to 0-1 range
        if accuracy_score is not None and accuracy_score > 1.0:
            accuracy_score = accuracy_score / 100.0
        
        # Calculate latency inverse (higher is better)
        latency_inverse = None
        if latency_seconds is not None and latency_seconds > 0:
            latency_inverse = 1.0 / latency_seconds
        
        eco_score = EcoEfficiencyScore(
            accuracy_score=accuracy_score,
            throughput_tokens_per_sec=throughput_tokens_per_sec,
            latency_inverse=latency_inverse,
            energy_kwh=impact.energy_kwh_with_pue,
            water_liters=impact.water_liters_total,
            carbon_kg=impact.carbon_kg
        )
        
        eco_score.calculate_efficiency()
        
        return eco_score
    
    def calculate_scaled_impact(
        self,
        per_query_impact: EnvironmentalImpact,
        queries_per_day: int,
        days: int = 365
    ) -> Tuple[EnvironmentalImpact, Dict[str, str]]:
        """
        Calculate scaled environmental impact for production deployment.
        
        Based on paper's GPT-4o case study: 700M queries/day.
        
        Args:
            per_query_impact: Environmental impact per single query
            queries_per_day: Number of queries per day
            days: Number of days to project (default: 365 for annual)
        
        Returns:
            Tuple of (scaled_impact, human_readable_equivalents)
        """
        total_queries = queries_per_day * days
        
        # Scale all metrics
        scaled_impact = EnvironmentalImpact(
            energy_kwh_raw=per_query_impact.energy_kwh_raw * total_queries,
            energy_kwh_with_pue=per_query_impact.energy_kwh_with_pue * total_queries,
            water_liters_site=per_query_impact.water_liters_site * total_queries,
            water_liters_source=per_query_impact.water_liters_source * total_queries,
            water_liters_total=per_query_impact.water_liters_total * total_queries,
            carbon_kg=per_query_impact.carbon_kg * total_queries,
            pue_overhead_kwh=per_query_impact.pue_overhead_kwh * total_queries,
            region=per_query_impact.region
        )
        
        # Calculate human-readable equivalents (from paper)
        energy_mwh = scaled_impact.energy_kwh_with_pue / 1000
        us_homes_equivalent = energy_mwh / 10.6  # Average US home: 10.6 MWh/year
        
        water_million_liters = scaled_impact.water_liters_total / 1_000_000
        people_drinking_water = water_million_liters / 0.365  # 365L per person per year
        
        carbon_metric_tons = scaled_impact.carbon_kg / 1000
        trees_needed = carbon_metric_tons * 36.4  # ~36.4 trees to offset 1 ton CO2/year
        chicago_forests = trees_needed / 3_600_000  # Chicago ~3.6M trees
        
        equivalents = {
            "Total Queries": f"{total_queries:,}",
            "Time Period": f"{days} days",
            "Energy": f"{energy_mwh:,.0f} MWh (equivalent to {us_homes_equivalent:,.0f} US homes)",
            "Water": f"{water_million_liters:.1f} million liters (drinking needs of {people_drinking_water:,.0f} people)",
            "Carbon": f"{carbon_metric_tons:,.0f} metric tons CO2e (requires {trees_needed:,.0f} trees to offset)",
            "Forest Size": f"{chicago_forests:.2f}x the size of Chicago's urban forest",
        }
        
        logger.info(f"Scaled impact calculated: {total_queries:,} queries over {days} days")
        logger.info(f"  Energy: {energy_mwh:,.0f} MWh ({us_homes_equivalent:,.0f} homes)")
        logger.info(f"  Water: {water_million_liters:.1f} ML ({people_drinking_water:,.0f} people)")
        logger.info(f"  Carbon: {carbon_metric_tons:,.0f} MT CO2e")
        
        return scaled_impact, equivalents
    
    def set_multipliers(self, multipliers: EnvironmentalMultipliers) -> None:
        """Update environmental multipliers."""
        self.multipliers = multipliers
        self.region = multipliers.region
        logger.info(f"Updated multipliers for region: {self.region.value}")
    
    def set_region(self, region: Region) -> None:
        """Update region and load default multipliers."""
        self.region = region
        self.multipliers = EnvironmentalMultipliers.get_regional_defaults(region)
        logger.info(f"Updated region to: {region.value}")
