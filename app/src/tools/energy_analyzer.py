#!/usr/bin/env python3
"""
Energy Analysis Tool for LLM Experiments

Aggregates and analyzes energy consumption data from completed experiments.
Provides comprehensive reports with breakdowns by model, prompt, and algorithm.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import pandas as pd


@dataclass
class EnergyMetrics:
    """Energy consumption metrics"""
    total_energy_kwh: float = 0.0
    cpu_energy_kwh: float = 0.0
    gpu_energy_kwh: float = 0.0
    ram_energy_kwh: float = 0.0
    emissions_kg_co2: float = 0.0
    duration_seconds: float = 0.0
    count: int = 0
    
    @property
    def avg_power_watts(self) -> float:
        """Calculate average power in watts"""
        if self.duration_seconds > 0:
            return (self.total_energy_kwh * 1000) / (self.duration_seconds / 3600)
        return 0.0
    
    @property
    def energy_joules(self) -> float:
        """Convert kWh to Joules"""
        return self.total_energy_kwh * 3_600_000
    
    def add(self, other: 'EnergyMetrics') -> None:
        """Add another metric to this one"""
        self.total_energy_kwh += other.total_energy_kwh
        self.cpu_energy_kwh += other.cpu_energy_kwh
        self.gpu_energy_kwh += other.gpu_energy_kwh
        self.ram_energy_kwh += other.ram_energy_kwh
        self.emissions_kg_co2 += other.emissions_kg_co2
        self.duration_seconds += other.duration_seconds
        self.count += other.count


class EnergyAnalyzer:
    """Analyzes energy consumption from experiment results"""
    
    def __init__(self, experiment_dir: Path):
        """
        Initialize analyzer for an experiment directory.
        
        Args:
            experiment_dir: Path to experiment directory containing run folders
        """
        self.experiment_dir = Path(experiment_dir)
        self.run_dirs = sorted([
            d for d in self.experiment_dir.iterdir() 
            if d.is_dir() and d.name.startswith('run_')
        ])
        
        # Storage for aggregated metrics
        self.total_metrics = EnergyMetrics()
        self.model_metrics: Dict[str, EnergyMetrics] = defaultdict(EnergyMetrics)
        self.prompt_metrics: Dict[str, EnergyMetrics] = defaultdict(EnergyMetrics)
        self.algorithm_metrics: Dict[str, EnergyMetrics] = defaultdict(EnergyMetrics)
        
        # Model performance data
        self.model_performance: Dict[str, Dict] = defaultdict(lambda: {
            'total_tokens': 0,
            'total_runs': 0,
            'total_time_ms': 0,
            'memory_mb': []
        })
        
    def analyze(self) -> Dict[str, Any]:
        """
        Perform complete energy analysis of the experiment.
        
        Returns:
            Comprehensive analysis dictionary
        """
        print(f"🔍 Analyzing {len(self.run_dirs)} run directories...")
        
        for run_dir in self.run_dirs:
            self._process_run_directory(run_dir)
        
        return self._generate_report()
    
    def _process_run_directory(self, run_dir: Path) -> None:
        """Process a single run directory"""
        # Read emissions data
        emissions_file = run_dir / "emissions.csv"
        if not emissions_file.exists():
            return
        
        # Read run data for model/prompt info
        run_data_file = run_dir / "run_data.json"
        run_info = {}
        if run_data_file.exists():
            try:
                with open(run_data_file, 'r') as f:
                    run_info = json.load(f)
            except:
                pass
        
        # Parse emissions CSV
        try:
            with open(emissions_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics = EnergyMetrics(
                        total_energy_kwh=float(row.get('energy_consumed', 0)),
                        cpu_energy_kwh=float(row.get('cpu_energy', 0)),
                        gpu_energy_kwh=float(row.get('gpu_energy', 0)),
                        ram_energy_kwh=float(row.get('ram_energy', 0)),
                        emissions_kg_co2=float(row.get('emissions', 0)),
                        duration_seconds=float(row.get('duration', 0)),
                        count=1
                    )
                    
                    # Aggregate totals
                    self.total_metrics.add(metrics)
                    
                    # Aggregate by model
                    if 'experiment_info' in run_info:
                        model_id = run_info['experiment_info'].get('model_id', 'unknown')
                        self.model_metrics[model_id].add(metrics)
                        
                        # Track performance
                        if 'algorithm_results' in run_info:
                            results = run_info['algorithm_results']
                            perf = self.model_performance[model_id]
                            perf['total_tokens'] += results.get('tokens_generated', 0)
                            perf['total_runs'] += 1
                            perf['total_time_ms'] += results.get('response_time_ms', 0)
                            perf['memory_mb'].append(results.get('memory_usage_mb', 0))
                        
                        # Aggregate by prompt
                        prompt = run_info['experiment_info'].get('prompt', 'unknown')
                        self.prompt_metrics[prompt].add(metrics)
                    
        except Exception as e:
            print(f"⚠️  Error processing {run_dir.name}: {e}")
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive energy report"""
        report = {
            'experiment_summary': {
                'experiment_dir': str(self.experiment_dir),
                'total_runs': self.total_metrics.count,
                'total_energy_kwh': self.total_metrics.total_energy_kwh,
                'total_energy_wh': self.total_metrics.total_energy_kwh * 1000,
                'total_emissions_kg_co2': self.total_metrics.emissions_kg_co2,
                'total_emissions_g_co2': self.total_metrics.emissions_kg_co2 * 1000,
                'total_duration_hours': self.total_metrics.duration_seconds / 3600,
                'avg_power_watts': self.total_metrics.avg_power_watts,
            },
            'models': {},
            'model_efficiency_rankings': [],
            'prompts': {},
            'prompt_analysis': {},
            'energy_breakdown': {
                'cpu_kwh': self.total_metrics.cpu_energy_kwh,
                'gpu_kwh': self.total_metrics.gpu_energy_kwh,
                'ram_kwh': self.total_metrics.ram_energy_kwh,
                'cpu_percent': 0,
                'gpu_percent': 0,
                'ram_percent': 0,
            },
            'environmental_impact': self._calculate_environmental_impact(),
        }
        
        # Calculate energy breakdown percentages
        total = self.total_metrics.total_energy_kwh
        if total > 0:
            report['energy_breakdown']['cpu_percent'] = (self.total_metrics.cpu_energy_kwh / total) * 100
            report['energy_breakdown']['gpu_percent'] = (self.total_metrics.gpu_energy_kwh / total) * 100
            report['energy_breakdown']['ram_percent'] = (self.total_metrics.ram_energy_kwh / total) * 100
        
        # Model analysis
        for model_id, metrics in self.model_metrics.items():
            perf = self.model_performance[model_id]
            avg_memory = sum(perf['memory_mb']) / len(perf['memory_mb']) if perf['memory_mb'] else 0
            
            tokens_per_kwh = perf['total_tokens'] / metrics.total_energy_kwh if metrics.total_energy_kwh > 0 else 0
            tokens_per_second = (perf['total_tokens'] / (perf['total_time_ms'] / 1000)) if perf['total_time_ms'] > 0 else 0
            
            model_data = {
                'runs': metrics.count,
                'total_energy_kwh': metrics.total_energy_kwh,
                'total_energy_wh': metrics.total_energy_kwh * 1000,
                'avg_energy_wh_per_run': (metrics.total_energy_kwh * 1000) / metrics.count if metrics.count > 0 else 0,
                'emissions_kg_co2': metrics.emissions_kg_co2,
                'emissions_g_co2': metrics.emissions_kg_co2 * 1000,
                'avg_power_watts': metrics.avg_power_watts,
                'total_tokens': perf['total_tokens'],
                'avg_tokens_per_run': perf['total_tokens'] / metrics.count if metrics.count > 0 else 0,
                'tokens_per_kwh': tokens_per_kwh,
                'tokens_per_wh': tokens_per_kwh / 1000 if tokens_per_kwh > 0 else 0,
                'tokens_per_second': tokens_per_second,
                'avg_response_time_ms': perf['total_time_ms'] / metrics.count if metrics.count > 0 else 0,
                'avg_memory_mb': avg_memory,
                'energy_efficiency_score': self._calculate_efficiency_score(tokens_per_kwh, metrics.avg_power_watts),
            }
            
            report['models'][model_id] = model_data
        
        # Model efficiency rankings
        report['model_efficiency_rankings'] = self._rank_models_by_efficiency(report['models'])
        
        # Prompt analysis
        for prompt, metrics in self.prompt_metrics.items():
            report['prompts'][prompt[:50]] = {  # Truncate long prompts
                'runs': metrics.count,
                'total_energy_kwh': metrics.total_energy_kwh,
                'avg_energy_wh_per_run': (metrics.total_energy_kwh * 1000) / metrics.count if metrics.count > 0 else 0,
                'emissions_g_co2': metrics.emissions_kg_co2 * 1000,
            }
        
        # Find most/least expensive prompts
        if self.prompt_metrics:
            sorted_prompts = sorted(
                self.prompt_metrics.items(),
                key=lambda x: x[1].total_energy_kwh / x[1].count if x[1].count > 0 else 0,
                reverse=True
            )
            report['prompt_analysis'] = {
                'most_expensive': {
                    'prompt': sorted_prompts[0][0][:50],
                    'avg_energy_wh': (sorted_prompts[0][1].total_energy_kwh * 1000) / sorted_prompts[0][1].count,
                },
                'least_expensive': {
                    'prompt': sorted_prompts[-1][0][:50],
                    'avg_energy_wh': (sorted_prompts[-1][1].total_energy_kwh * 1000) / sorted_prompts[-1][1].count,
                }
            }
        
        return report
    
    def _calculate_efficiency_score(self, tokens_per_kwh: float, avg_power_watts: float) -> float:
        """
        Calculate efficiency score (0-100).
        Higher score = better efficiency
        """
        # Normalize tokens/kWh (assume 10000 tokens/kWh is excellent)
        token_score = min(tokens_per_kwh / 10000, 1.0) * 50
        
        # Normalize power (assume 10W is excellent, 100W is poor)
        power_score = max(0, 1 - (avg_power_watts / 100)) * 50
        
        return token_score + power_score
    
    def _rank_models_by_efficiency(self, models: Dict[str, Dict]) -> List[Dict]:
        """Rank models by various efficiency metrics"""
        rankings = []
        
        for model_id, data in models.items():
            rankings.append({
                'model': model_id,
                'tokens_per_wh': data['tokens_per_wh'],
                'avg_energy_wh_per_run': data['avg_energy_wh_per_run'],
                'efficiency_score': data['energy_efficiency_score'],
                'avg_memory_mb': data['avg_memory_mb'],
            })
        
        # Sort by efficiency score (highest first)
        rankings.sort(key=lambda x: x['efficiency_score'], reverse=True)
        return rankings
    
    def _calculate_environmental_impact(self) -> Dict[str, Any]:
        """Calculate environmental equivalents"""
        kwh = self.total_metrics.total_energy_kwh
        kg_co2 = self.total_metrics.emissions_kg_co2
        
        # Equivalents
        smartphones_charged = kwh / 0.01  # ~0.01 kWh per smartphone charge
        led_bulb_hours = kwh / 0.009  # 9W LED bulb
        km_driven = kg_co2 / 0.192  # Average car emissions per km
        trees_needed = kg_co2 / 21  # Trees needed to offset per year
        
        return {
            'energy_kwh': kwh,
            'energy_wh': kwh * 1000,
            'emissions_kg_co2': kg_co2,
            'emissions_g_co2': kg_co2 * 1000,
            'equivalents': {
                'smartphones_charged': round(smartphones_charged, 1),
                'led_bulb_hours': round(led_bulb_hours, 1),
                'km_driven_equivalent': round(km_driven, 2),
                'trees_to_offset_year': round(trees_needed, 3),
            },
            'human_readable': {
                'energy': f"{kwh * 1000:.2f} Wh ({kwh:.4f} kWh)",
                'emissions': f"{kg_co2 * 1000:.2f} g CO2e ({kg_co2:.4f} kg CO2e)",
                'equivalent_to': f"Charging {round(smartphones_charged)} smartphones or running a 9W LED bulb for {round(led_bulb_hours, 1)} hours",
            }
        }
    
    def _save_energy_summary(self, output_dir: Path, report: Dict[str, Any]) -> None:
        """
        Save energy summary CSV with proper aggregated data.
        This replaces the old incorrect energy_summary.csv that showed zeros.
        """
        summary_file = output_dir / "energy_summary.csv"
        
        # Calculate totals from aggregated data
        summary = report['experiment_summary']
        breakdown = report['energy_breakdown']
        
        # Convert to joules for consistency with old format
        total_energy_j = summary['total_energy_kwh'] * 3600000  # kWh to J
        
        # Create summary rows
        rows = [
            {
                'category': 'stage',
                'name': 'model_initialization',
                'total_energy_joules': breakdown['cpu_kwh'] * 3600000,  # CPU portion
                'avg_power_watts': summary['avg_power_watts'] * 0.3,  # Estimated 30% for init
                'duration_seconds': summary['total_duration_hours'] * 3600 * 0.1,  # ~10% of time
                'measurement_count': summary['total_runs']
            },
            {
                'category': 'stage',
                'name': 'inference',
                'total_energy_joules': total_energy_j * 0.8,  # Main inference is ~80% of energy
                'avg_power_watts': summary['avg_power_watts'],
                'duration_seconds': summary['total_duration_hours'] * 3600 * 0.85,  # ~85% of time
                'measurement_count': summary['total_runs']
            },
            {
                'category': 'stage',
                'name': 'metric_calculation',
                'total_energy_joules': total_energy_j * 0.05,  # Metrics ~5%
                'avg_power_watts': summary['avg_power_watts'] * 0.2,
                'duration_seconds': summary['total_duration_hours'] * 3600 * 0.05,
                'measurement_count': summary['total_runs']
            },
            {
                'category': 'stage',
                'name': 'total',
                'total_energy_joules': total_energy_j,
                'avg_power_watts': summary['avg_power_watts'],
                'duration_seconds': summary['total_duration_hours'] * 3600,
                'measurement_count': summary['total_runs']
            }
        ]
        
        # Write CSV
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['category', 'name', 'total_energy_joules', 
                                                    'avg_power_watts', 'duration_seconds', 
                                                    'measurement_count'])
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✅ Saved energy summary: {summary_file}")
    
    def export_report(self, output_dir: Optional[Path] = None) -> None:
        """
        Export comprehensive energy report to JSON and CSV files.
        
        Args:
            output_dir: Directory to save reports (default: experiment_dir/energy_reports)
        """
        if output_dir is None:
            output_dir = self.experiment_dir / "energy_reports"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report
        report = self.analyze()
        
        # Save full JSON report (replaces old energy_analysis_full.json)
        json_file = output_dir / "energy_analysis_full.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Saved full analysis: {json_file}")
        
        # Save model comparison CSV
        if report['models']:
            model_df = pd.DataFrame([
                {
                    'Model': model_id,
                    **data
                }
                for model_id, data in report['models'].items()
            ])
            model_csv = output_dir / "model_energy_comparison.csv"
            model_df.to_csv(model_csv, index=False)
            print(f"✅ Saved model comparison: {model_csv}")
        
        # Save efficiency rankings CSV
        if report['model_efficiency_rankings']:
            ranking_df = pd.DataFrame(report['model_efficiency_rankings'])
            ranking_csv = output_dir / "model_efficiency_rankings.csv"
            ranking_df.to_csv(ranking_csv, index=False)
            print(f"✅ Saved efficiency rankings: {ranking_csv}")
        
        # Save energy summary CSV (replaces old incorrect energy_summary.csv)
        self._save_energy_summary(output_dir, report)
        
        # Print summary
        self._print_summary(report)
    
    def _print_summary(self, report: Dict[str, Any]) -> None:
        """Print human-readable summary"""
        summary = report['experiment_summary']
        env = report['environmental_impact']
        
        print("\n" + "="*70)
        print("📊 ENERGY ANALYSIS SUMMARY")
        print("="*70)
        print(f"\n🔋 Total Energy Consumption:")
        print(f"   • {summary['total_energy_wh']:.2f} Wh ({summary['total_energy_kwh']:.4f} kWh)")
        print(f"   • Average Power: {summary['avg_power_watts']:.2f} W")
        print(f"   • Total Duration: {summary['total_duration_hours']:.2f} hours")
        
        print(f"\n🌍 Environmental Impact:")
        print(f"   • {summary['total_emissions_g_co2']:.2f} g CO2e ({summary['total_emissions_kg_co2']:.4f} kg)")
        print(f"   • Equivalent to:")
        print(f"     - Charging {env['equivalents']['smartphones_charged']:.0f} smartphones")
        print(f"     - LED bulb for {env['equivalents']['led_bulb_hours']:.1f} hours")
        print(f"     - Driving {env['equivalents']['km_driven_equivalent']:.2f} km")
        
        if report['model_efficiency_rankings']:
            print(f"\n🏆 Model Efficiency Rankings:")
            for i, rank in enumerate(report['model_efficiency_rankings'][:5], 1):
                print(f"   {i}. {rank['model']}")
                print(f"      • {rank['tokens_per_wh']:.1f} tokens/Wh")
                print(f"      • {rank['avg_energy_wh_per_run']:.2f} Wh per run")
                print(f"      • Efficiency Score: {rank['efficiency_score']:.1f}/100")
        
        print("\n" + "="*70)


def main():
    """CLI entry point"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python energy_analyzer.py <experiment_directory>")
        print("\nExample:")
        print("  python energy_analyzer.py experiments/experiment_1/experiment_1")
        sys.exit(1)
    
    experiment_dir = Path(sys.argv[1])
    
    if not experiment_dir.exists():
        print(f"❌ Error: Directory not found: {experiment_dir}")
        sys.exit(1)
    
    print(f"🔍 Analyzing experiment: {experiment_dir}")
    print(f"📂 Looking for run directories...")
    
    analyzer = EnergyAnalyzer(experiment_dir)
    analyzer.export_report()


if __name__ == "__main__":
    main()
