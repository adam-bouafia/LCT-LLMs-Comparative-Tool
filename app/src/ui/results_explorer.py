#!/usr/bin/env python3
"""
Results Explorer for LLM Experiment Runner

This tool allows you to explore, analyze, and manage experiment results.
Features:
- View experiment results
- Delete experiments
- Compare experiments
- Export results
"""

import os
import sys
import json
import csv
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text


class ResultsExplorer:
    def __init__(self):
        self.console = Console()
        self.experiments_dir = Path("experiments")
        
    def show_header(self):
        """Display the tool header."""
        self.console.print()
        self.console.print(Panel.fit(
            "[bold blue]🔍 LLM Experiment Results Explorer[/bold blue]\n"
            "[dim]Explore, analyze, and manage your experiment results[/dim]",
            border_style="blue"
        ))
        self.console.print()

    def scan_experiments(self) -> List[Dict]:
        """Scan for available experiments."""
        experiments = []
        
        if not self.experiments_dir.exists():
            return experiments
            
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
                exp_info = self.get_experiment_info(exp_dir)
                if exp_info:
                    experiments.append(exp_info)
                    
        return sorted(experiments, key=lambda x: x.get('last_modified', ''), reverse=True)

    def get_experiment_info(self, exp_dir: Path) -> Optional[Dict]:
        """Get information about a single experiment."""
        try:
            info = {
                'name': exp_dir.name,
                'path': str(exp_dir),
                'last_modified': datetime.fromtimestamp(exp_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'size': self.get_dir_size(exp_dir),
                'runs': 0,
                'csv_exists': False,
                'json_exists': False,
                'config_exists': False,
                'has_results': False
            }
            
            # Check for CSV results
            csv_file = exp_dir / 'run_table.csv'
            if csv_file.exists():
                info['csv_exists'] = True
                info['has_results'] = True
                info['runs'] = self.count_csv_runs(csv_file)
            
            # Check for JSON results    
            json_files = list(exp_dir.glob('*.json'))
            if json_files:
                info['json_exists'] = True
                info['has_results'] = True
                
            # Check for config file
            config_file = exp_dir / 'RunnerConfig.py'
            if config_file.exists():
                info['config_exists'] = True
                
            # Count run directories
            run_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
            if run_dirs:
                info['runs'] = max(info['runs'], len(run_dirs))
                info['has_results'] = True
                
            return info
            
        except Exception as e:
            self.console.print(f"[red]Error scanning {exp_dir.name}: {e}[/red]")
            return None

    def count_csv_runs(self, csv_file: Path) -> int:
        """Count the number of runs in a CSV file."""
        try:
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                return max(0, sum(1 for row in reader) - 1)  # Subtract header
        except:
            return 0

    def get_dir_size(self, path: Path) -> str:
        """Get human-readable directory size."""
        try:
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            
            for unit in ['B', 'KB', 'MB', 'GB']:
                if total_size < 1024:
                    return f"{total_size:.1f}{unit}"
                total_size /= 1024
            return f"{total_size:.1f}TB"
        except:
            return "Unknown"

    def show_experiments_table(self, experiments: List[Dict]):
        """Display experiments in a table."""
        if not experiments:
            self.console.print("[yellow]No experiments found.[/yellow]")
            return
            
        table = Table(title="Available Experiments")
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Name", style="bold")
        table.add_column("Last Modified", style="dim")
        table.add_column("Runs", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Files", style="green")
        table.add_column("Status")
        
        for i, exp in enumerate(experiments, 1):
            files = []
            if exp['csv_exists']:
                files.append("CSV")
            if exp['json_exists']:
                files.append("JSON")
            if exp['config_exists']:
                files.append("Config")
                
            status = "✅ Complete" if exp['has_results'] else "⚪ Empty"
            
            table.add_row(
                str(i),
                exp['name'],
                exp['last_modified'],
                str(exp['runs']),
                exp['size'],
                ", ".join(files) if files else "None",
                status
            )
        
        self.console.print(table)

    def view_experiment_details(self, exp_info: Dict):
        """Show detailed information about an experiment."""
        exp_dir = Path(exp_info['path'])
        
        self.console.print(f"\n[bold blue]📊 Experiment: {exp_info['name']}[/bold blue]")
        self.console.print("─" * 50)
        
        # Basic info
        info_table = Table(show_header=False, box=None)
        info_table.add_column("Field", style="cyan")
        info_table.add_column("Value")
        
        info_table.add_row("Path", exp_info['path'])
        info_table.add_row("Last Modified", exp_info['last_modified'])
        info_table.add_row("Size", exp_info['size'])
        info_table.add_row("Total Runs", str(exp_info['runs']))
        
        self.console.print(info_table)
        
        # Show CSV results if available
        csv_file = exp_dir / 'run_table.csv'
        if csv_file.exists():
            self.console.print("\n[green]📋 CSV Results Preview:[/green]")
            self.show_csv_preview(csv_file)
            
            # Add option to view full CSV content
            if Confirm.ask("\n[bold cyan]Would you like to view the full CSV content?[/bold cyan]", default=False):
                self.view_csv_content(csv_file)
        
        # Show run directories
        run_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
        if run_dirs:
            self.console.print(f"\n[cyan]📁 Run Directories ({len(run_dirs)}):[/cyan]")
            for run_dir in run_dirs[:10]:  # Show first 10
                emissions_file = run_dir / 'emissions.csv'
                status = "✅" if emissions_file.exists() else "⚪"
                self.console.print(f"  {status} {run_dir.name}")
            if len(run_dirs) > 10:
                self.console.print(f"  ... and {len(run_dirs) - 10} more")

    def show_csv_preview(self, csv_file: Path):
        """Show a preview of CSV results."""
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            if not rows:
                self.console.print("  [dim]No data in CSV file[/dim]")
                return
                
            # Show summary stats
            completed_runs = sum(1 for row in rows if row.get('__done', '').upper() == 'DONE')
            self.console.print(f"  Total rows: {len(rows)}")
            self.console.print(f"  Completed runs: {completed_runs}")
            
            # Show columns
            if rows:
                columns = list(rows[0].keys())
                self.console.print(f"  Columns: {', '.join(columns[:5])}" + 
                                 (f" ... +{len(columns)-5}" if len(columns) > 5 else ""))
            
        except Exception as e:
            self.console.print(f"  [red]Error reading CSV: {e}[/red]")

    def view_csv_content(self, csv_file: Path):
        """Display full CSV content in an interactive viewer."""
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            if not rows:
                self.console.print("[yellow]No data in CSV file[/yellow]")
                return
                
            columns = list(rows[0].keys())
            
            while True:
                self.console.clear()
                self.console.print(f"[bold blue]📋 CSV Content: {csv_file.name}[/bold blue]")
                self.console.print(f"Rows: {len(rows)} | Columns: {len(columns)}")
                self.console.print("─" * 80)
                
                # Show CSV viewing options
                self.console.print("\n[bold]View Options:[/bold]")
                self.console.print("1. Show all data (table format)")
                self.console.print("2. Show specific columns")
                self.console.print("3. Filter by status/completion")
                self.console.print("4. Show statistics summary")
                self.console.print("5. Search in data")
                self.console.print("6. Export filtered data")
                self.console.print("0. Back to experiment details")
                
                choice = Prompt.ask("\nChoice", choices=['0', '1', '2', '3', '4', '5', '6'])
                
                if choice == '0':
                    break
                elif choice == '1':
                    self.show_full_csv_table(rows, columns)
                elif choice == '2':
                    self.show_selected_columns(rows, columns)
                elif choice == '3':
                    self.filter_by_status(rows, columns)
                elif choice == '4':
                    self.show_csv_statistics(rows, columns)
                elif choice == '5':
                    self.search_csv_data(rows, columns)
                elif choice == '6':
                    self.export_filtered_csv(csv_file, rows, columns)
                    
        except Exception as e:
            self.console.print(f"[red]❌ Error reading CSV: {e}[/red]")

    def show_full_csv_table(self, rows: List[Dict], columns: List[str]):
        """Display the full CSV data in table format."""
        # Limit columns for display readability
        display_columns = columns[:8] if len(columns) > 8 else columns
        
        table = Table(title="CSV Data", show_lines=True)
        
        for col in display_columns:
            # Truncate long column names
            col_name = col if len(col) <= 15 else col[:12] + "..."
            table.add_column(col_name, overflow="fold", max_width=20)
            
        # Add rows (limit to prevent overwhelming display)
        display_rows = rows[:50] if len(rows) > 50 else rows
        
        for row in display_rows:
            row_data = []
            for col in display_columns:
                value = str(row.get(col, ''))
                # Truncate long values
                if len(value) > 25:
                    value = value[:22] + "..."
                row_data.append(value)
            table.add_row(*row_data)
            
        self.console.print(table)
        
        if len(columns) > 8:
            self.console.print(f"\n[dim]Note: Showing {len(display_columns)} of {len(columns)} columns[/dim]")
        if len(rows) > 50:
            self.console.print(f"[dim]Note: Showing {len(display_rows)} of {len(rows)} rows[/dim]")
            
        Prompt.ask("\nPress Enter to continue")

    def show_selected_columns(self, rows: List[Dict], columns: List[str]):
        """Allow user to select specific columns to view."""
        self.console.print("\n[bold]Available Columns:[/bold]")
        for i, col in enumerate(columns, 1):
            self.console.print(f"{i:2d}. {col}")
            
        selection = Prompt.ask(
            "\nEnter column numbers (e.g., 1,3,5) or 'all'", 
            default="all"
        )
        
        if selection.lower() == 'all':
            selected_columns = columns
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected_columns = [columns[i] for i in indices if 0 <= i < len(columns)]
            except:
                self.console.print("[red]Invalid selection[/red]")
                return
                
        if not selected_columns:
            self.console.print("[red]No valid columns selected[/red]")
            return
            
        # Display selected columns
        table = Table(title=f"Selected Columns ({len(selected_columns)})")
        
        for col in selected_columns:
            table.add_column(col, overflow="fold", max_width=25)
            
        for row in rows[:30]:  # Limit rows for readability
            row_data = [str(row.get(col, '')) for col in selected_columns]
            table.add_row(*row_data)
            
        self.console.print(table)
        if len(rows) > 30:
            self.console.print(f"[dim]Showing first 30 of {len(rows)} rows[/dim]")
            
        Prompt.ask("\nPress Enter to continue")

    def filter_by_status(self, rows: List[Dict], columns: List[str]):
        """Filter data by completion status."""
        # Find status-like columns
        status_columns = [col for col in columns if 'done' in col.lower() or 'status' in col.lower()]
        
        if not status_columns:
            self.console.print("[yellow]No status columns found[/yellow]")
            return
            
        status_col = status_columns[0]  # Use first status column
        
        # Get unique values in status column
        unique_values = list(set(str(row.get(status_col, '')) for row in rows))
        
        self.console.print(f"\n[bold]Filter by {status_col}:[/bold]")
        for i, value in enumerate(unique_values, 1):
            count = sum(1 for row in rows if str(row.get(status_col, '')) == value)
            self.console.print(f"{i}. {value} ({count} rows)")
            
        selection = Prompt.ask("Select filter value", choices=[str(i) for i in range(1, len(unique_values) + 1)])
        
        try:
            filter_value = unique_values[int(selection) - 1]
            filtered_rows = [row for row in rows if str(row.get(status_col, '')) == filter_value]
            
            self.console.print(f"\n[green]Filtered to {len(filtered_rows)} rows where {status_col} = {filter_value}[/green]")
            
            # Show filtered data
            if filtered_rows:
                display_columns = columns[:6]  # Limit columns
                table = Table(title=f"Filtered Data: {status_col} = {filter_value}")
                
                for col in display_columns:
                    table.add_column(col, overflow="fold", max_width=20)
                    
                for row in filtered_rows[:20]:  # Limit rows
                    row_data = [str(row.get(col, '')) for col in display_columns]
                    table.add_row(*row_data)
                    
                self.console.print(table)
                if len(filtered_rows) > 20:
                    self.console.print(f"[dim]Showing first 20 of {len(filtered_rows)} filtered rows[/dim]")
                    
        except:
            self.console.print("[red]Invalid selection[/red]")
            
        Prompt.ask("\nPress Enter to continue")

    def show_csv_statistics(self, rows: List[Dict], columns: List[str]):
        """Show statistical summary of CSV data."""
        self.console.print("\n[bold blue]📊 CSV Statistics[/bold blue]")
        
        stats_table = Table(title="Data Overview")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right")
        
        stats_table.add_row("Total Rows", str(len(rows)))
        stats_table.add_row("Total Columns", str(len(columns)))
        
        # Status analysis
        status_cols = [col for col in columns if 'done' in col.lower()]
        if status_cols and rows:
            status_col = status_cols[0]
            completed = sum(1 for row in rows if str(row.get(status_col, '')).upper() == 'DONE')
            stats_table.add_row("Completed Runs", f"{completed} ({completed/len(rows)*100:.1f}%)")
            
        # Numeric column analysis
        numeric_cols = []
        for col in columns:
            if any(str(row.get(col, '')).replace('.', '').replace('-', '').isdigit() 
                  for row in rows[:5] if row.get(col)):
                numeric_cols.append(col)
                
        if numeric_cols:
            stats_table.add_row("Numeric Columns", str(len(numeric_cols)))
            
        self.console.print(stats_table)
        
        # Show column details
        if numeric_cols:
            self.console.print("\n[bold]Numeric Column Analysis:[/bold]")
            for col in numeric_cols[:5]:  # Show first 5 numeric columns
                values = []
                for row in rows:
                    try:
                        val = float(row.get(col, 0))
                        values.append(val)
                    except:
                        continue
                        
                if values:
                    self.console.print(f"  {col}: avg={sum(values)/len(values):.2f}, min={min(values):.2f}, max={max(values):.2f}")
                    
        Prompt.ask("\nPress Enter to continue")

    def search_csv_data(self, rows: List[Dict], columns: List[str]):
        """Search for specific text in CSV data."""
        search_term = Prompt.ask("\nEnter search term")
        
        if not search_term:
            return
            
        # Search in all columns
        matching_rows = []
        for row in rows:
            for col, value in row.items():
                if search_term.lower() in str(value).lower():
                    matching_rows.append(row)
                    break
                    
        self.console.print(f"\n[green]Found {len(matching_rows)} rows containing '{search_term}'[/green]")
        
        if matching_rows:
            # Show matching data
            display_columns = columns[:6]
            table = Table(title=f"Search Results: '{search_term}'")
            
            for col in display_columns:
                table.add_column(col, overflow="fold", max_width=20)
                
            for row in matching_rows[:15]:  # Limit display
                row_data = []
                for col in display_columns:
                    value = str(row.get(col, ''))
                    # Highlight search term
                    if search_term.lower() in value.lower():
                        value = value.replace(search_term, f"[bold red]{search_term}[/bold red]")
                    row_data.append(value)
                table.add_row(*row_data)
                
            self.console.print(table)
            if len(matching_rows) > 15:
                self.console.print(f"[dim]Showing first 15 of {len(matching_rows)} matching rows[/dim]")
                
        Prompt.ask("\nPress Enter to continue")

    def export_filtered_csv(self, original_file: Path, rows: List[Dict], columns: List[str]):
        """Export filtered CSV data to a new file."""
        self.console.print("\n[bold]Export Options:[/bold]")
        self.console.print("1. Export all data")
        self.console.print("2. Export completed runs only")
        self.console.print("3. Export selected columns only")
        self.console.print("0. Cancel")
        
        choice = Prompt.ask("Choice", choices=['0', '1', '2', '3'])
        
        if choice == '0':
            return
            
        export_rows = rows
        export_columns = columns
        suffix = ""
        
        if choice == '2':
            # Filter completed runs
            status_cols = [col for col in columns if 'done' in col.lower()]
            if status_cols:
                status_col = status_cols[0]
                export_rows = [row for row in rows if str(row.get(status_col, '')).upper() == 'DONE']
                suffix = "_completed"
                
        elif choice == '3':
            # Select columns
            self.console.print("\nSelect columns to export (comma-separated numbers):")
            for i, col in enumerate(columns, 1):
                self.console.print(f"{i:2d}. {col}")
                
            selection = Prompt.ask("Column numbers")
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                export_columns = [columns[i] for i in indices if 0 <= i < len(columns)]
                suffix = "_filtered"
            except:
                self.console.print("[red]Invalid selection[/red]")
                return
                
        # Create export filename
        base_name = original_file.stem
        export_file = Path(f"{base_name}{suffix}_export.csv")
        
        try:
            with open(export_file, 'w', newline='') as f:
                if export_rows:
                    writer = csv.DictWriter(f, fieldnames=export_columns)
                    writer.writeheader()
                    for row in export_rows:
                        filtered_row = {col: row.get(col, '') for col in export_columns}
                        writer.writerow(filtered_row)
                        
            self.console.print(f"[green]✅ Exported {len(export_rows)} rows to {export_file}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Export failed: {e}[/red]")
            
        Prompt.ask("\nPress Enter to continue")

    def delete_all_experiments(self, experiments: List[Dict]):
        """Delete all experiments with confirmation."""
        if not experiments:
            self.console.print("[yellow]No experiments to delete.[/yellow]")
            return
            
        self.console.print(f"\n[bold red]⚠️  Delete All Experiments ({len(experiments)} total)[/bold red]")
        self.console.print("This will permanently delete:")
        
        # Show summary of what will be deleted
        total_size = 0
        total_runs = 0
        experiments_with_data = 0
        
        for exp in experiments:
            if exp['has_results']:
                experiments_with_data += 1
            total_runs += exp['runs']
            # Convert size string back to bytes for calculation
            try:
                size_str = exp['size']
                if 'KB' in size_str:
                    total_size += float(size_str.replace('KB', '')) * 1024
                elif 'MB' in size_str:
                    total_size += float(size_str.replace('MB', '')) * 1024 * 1024
                elif 'GB' in size_str:
                    total_size += float(size_str.replace('GB', '')) * 1024 * 1024 * 1024
            except:
                pass
        
        # Convert total size back to human readable
        if total_size < 1024:
            size_display = f"{total_size:.1f}B"
        elif total_size < 1024**2:
            size_display = f"{total_size/1024:.1f}KB"
        elif total_size < 1024**3:
            size_display = f"{total_size/(1024**2):.1f}MB"
        else:
            size_display = f"{total_size/(1024**3):.1f}GB"
            
        summary_table = Table(title="Deletion Summary", show_header=False)
        summary_table.add_column("Item", style="cyan")
        summary_table.add_column("Count", justify="right")
        
        summary_table.add_row("Total Experiments", str(len(experiments)))
        summary_table.add_row("Experiments with Results", str(experiments_with_data))
        summary_table.add_row("Total Runs", str(total_runs))
        summary_table.add_row("Total Size", size_display)
        
        self.console.print(summary_table)
        
        # Multiple confirmations for safety
        if not Confirm.ask(f"\n[red]Are you sure you want to delete ALL {len(experiments)} experiments?[/red]"):
            self.console.print("[yellow]Deletion cancelled.[/yellow]")
            return
            
        if experiments_with_data > 0:
            if not Confirm.ask(f"[red]{experiments_with_data} experiments contain results data. Really delete?[/red]"):
                self.console.print("[yellow]Deletion cancelled.[/yellow]")
                return
                
        # Option to create backup archive
        backup_created = False
        if Confirm.ask("Create backup archive before deleting all experiments?"):
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                archive_name = f"experiments_backup_{timestamp}"
                
                self.console.print("[yellow]Creating backup archive...[/yellow]")
                shutil.make_archive(archive_name, 'zip', self.experiments_dir)
                backup_created = True
                self.console.print(f"[green]✅ Backup created: {archive_name}.zip[/green]")
                
            except Exception as e:
                self.console.print(f"[red]❌ Backup failed: {e}[/red]")
                if not Confirm.ask("Continue with deletion without backup?"):
                    self.console.print("[yellow]Deletion cancelled.[/yellow]")
                    return
        
        # Final confirmation
        final_msg = "FINAL CONFIRMATION: Delete all experiments?"
        if backup_created:
            final_msg += " (backup created)"
            
        if not Confirm.ask(f"[bold red]{final_msg}[/bold red]"):
            self.console.print("[yellow]Deletion cancelled.[/yellow]")
            return
            
        # Perform deletions
        deleted_count = 0
        failed_deletions = []
        
        with self.console.status("[red]Deleting experiments...", spinner="dots"):
            for exp in experiments:
                try:
                    exp_path = Path(exp['path'])
                    if exp_path.exists():
                        shutil.rmtree(exp_path)
                        deleted_count += 1
                except Exception as e:
                    failed_deletions.append((exp['name'], str(e)))
        
        # Report results
        if deleted_count == len(experiments):
            self.console.print(f"[green]✅ Successfully deleted all {deleted_count} experiments![/green]")
        else:
            self.console.print(f"[yellow]⚠️  Deleted {deleted_count} of {len(experiments)} experiments[/yellow]")
            
            if failed_deletions:
                self.console.print("\n[red]Failed deletions:[/red]")
                for name, error in failed_deletions:
                    self.console.print(f"  • {name}: {error}")
                    
        Prompt.ask("\nPress Enter to continue")

    def export_all_experiments(self, experiments: List[Dict]):
        """Export all experiments to various formats."""
        if not experiments:
            self.console.print("[yellow]No experiments to export.[/yellow]")
            return
            
        # Filter experiments that have data to export
        exportable_experiments = [exp for exp in experiments if exp['has_results']]
        
        self.console.print(f"\n[bold blue]📤 Export All Experiments[/bold blue]")
        self.console.print(f"Found {len(exportable_experiments)} experiments with exportable data")
        
        if not exportable_experiments:
            self.console.print("[yellow]No experiments contain exportable data (CSV/JSON files).[/yellow]")
            return
            
        # Show what will be exported
        export_table = Table(title="Exportable Experiments")
        export_table.add_column("Name", style="bold")
        export_table.add_column("Files", style="green")
        export_table.add_column("Runs", justify="right")
        export_table.add_column("Size")
        
        for exp in exportable_experiments:
            files = []
            if exp['csv_exists']:
                files.append("CSV")
            if exp['json_exists']:
                files.append("JSON")
            if exp['config_exists']:
                files.append("Config")
                
            export_table.add_row(
                exp['name'],
                ", ".join(files),
                str(exp['runs']),
                exp['size']
            )
            
        self.console.print(export_table)
        
        # Export options
        self.console.print("\n[bold]Export Options:[/bold]")
        self.console.print("1. Create individual archives for each experiment")
        self.console.print("2. Create one combined archive with all experiments")
        self.console.print("3. Export all CSV files to current directory")
        self.console.print("4. Create comprehensive summary report (JSON)")
        self.console.print("5. Export everything (archives + CSVs + summary)")
        self.console.print("0. Cancel")
        
        choice = Prompt.ask("Choose export option", choices=['0', '1', '2', '3', '4', '5'])
        
        if choice == '0':
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_results = []
        
        try:
            if choice in ['1', '5']:
                # Individual archives
                self.console.print("[yellow]Creating individual archives...[/yellow]")
                with self.console.status("[blue]Creating archives...", spinner="dots"):
                    for exp in exportable_experiments:
                        try:
                            archive_name = f"{exp['name']}_export_{timestamp}"
                            shutil.make_archive(archive_name, 'zip', exp['path'])
                            export_results.append(f"✅ {archive_name}.zip")
                        except Exception as e:
                            export_results.append(f"❌ {exp['name']}: {e}")
                            
            if choice in ['2', '5']:
                # Combined archive
                self.console.print("[yellow]Creating combined archive...[/yellow]")
                try:
                    combined_name = f"all_experiments_{timestamp}"
                    shutil.make_archive(combined_name, 'zip', self.experiments_dir)
                    export_results.append(f"✅ {combined_name}.zip (combined)")
                except Exception as e:
                    export_results.append(f"❌ Combined archive: {e}")
                    
            if choice in ['3', '5']:
                # Export all CSV files
                self.console.print("[yellow]Exporting CSV files...[/yellow]")
                csv_count = 0
                for exp in exportable_experiments:
                    if exp['csv_exists']:
                        try:
                            csv_source = Path(exp['path']) / 'run_table.csv'
                            csv_dest = Path(f"{exp['name']}_results_{timestamp}.csv")
                            shutil.copy2(csv_source, csv_dest)
                            csv_count += 1
                            export_results.append(f"✅ {csv_dest}")
                        except Exception as e:
                            export_results.append(f"❌ {exp['name']} CSV: {e}")
                export_results.append(f"📊 Exported {csv_count} CSV files")
                
            if choice in ['4', '5']:
                # Comprehensive summary
                self.console.print("[yellow]Creating summary report...[/yellow]")
                try:
                    summary = self.create_comprehensive_summary(exportable_experiments)
                    summary_file = Path(f"experiments_summary_{timestamp}.json")
                    with open(summary_file, 'w') as f:
                        json.dump(summary, f, indent=2, default=str)
                    export_results.append(f"✅ {summary_file}")
                except Exception as e:
                    export_results.append(f"❌ Summary report: {e}")
                    
            # Show results
            self.console.print("\n[bold green]📋 Export Results:[/bold green]")
            for result in export_results:
                self.console.print(f"  {result}")
                
        except Exception as e:
            self.console.print(f"[red]❌ Export failed: {e}[/red]")
            
        Prompt.ask("\nPress Enter to continue")

    def create_comprehensive_summary(self, experiments: List[Dict]) -> Dict:
        """Create a comprehensive summary of all experiments."""
        summary = {
            'export_info': {
                'generated_at': datetime.now().isoformat(),
                'total_experiments': len(experiments),
                'export_tool': 'LLM Experiment Results Explorer'
            },
            'experiments': [],
            'aggregate_stats': {
                'total_runs': 0,
                'total_completed_runs': 0,
                'experiments_with_csv': 0,
                'experiments_with_json': 0,
                'experiments_with_config': 0
            }
        }
        
        for exp in experiments:
            exp_summary = {
                'name': exp['name'],
                'last_modified': exp['last_modified'],
                'size': exp['size'],
                'runs': exp['runs'],
                'files': {
                    'csv': exp['csv_exists'],
                    'json': exp['json_exists'],
                    'config': exp['config_exists']
                },
                'has_results': exp['has_results']
            }
            
            # Try to get CSV data if available
            if exp['csv_exists']:
                try:
                    csv_file = Path(exp['path']) / 'run_table.csv'
                    with open(csv_file, 'r') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        
                    exp_summary['csv_analysis'] = {
                        'total_rows': len(rows),
                        'columns': list(rows[0].keys()) if rows else [],
                        'completed_runs': sum(1 for row in rows if row.get('__done', '').upper() == 'DONE')
                    }
                    
                    # Update aggregate stats
                    summary['aggregate_stats']['total_runs'] += len(rows)
                    summary['aggregate_stats']['total_completed_runs'] += exp_summary['csv_analysis']['completed_runs']
                    
                except Exception as e:
                    exp_summary['csv_error'] = str(e)
                    
            summary['experiments'].append(exp_summary)
            
            # Update file type counts
            if exp['csv_exists']:
                summary['aggregate_stats']['experiments_with_csv'] += 1
            if exp['json_exists']:
                summary['aggregate_stats']['experiments_with_json'] += 1  
            if exp['config_exists']:
                summary['aggregate_stats']['experiments_with_config'] += 1
                
        return summary

    def delete_experiment(self, exp_info: Dict):
        """Delete an experiment with confirmation."""
        exp_name = exp_info['name']
        exp_path = Path(exp_info['path'])
        
        self.console.print(f"\n[bold red]⚠️  Delete Experiment: {exp_name}[/bold red]")
        self.console.print(f"Path: {exp_path}")
        self.console.print(f"Size: {exp_info['size']}")
        self.console.print(f"Runs: {exp_info['runs']}")
        
        if not Confirm.ask(f"\n[red]Are you sure you want to delete '{exp_name}'? This cannot be undone.[/red]"):
            self.console.print("[yellow]Deletion cancelled.[/yellow]")
            return
            
        # Double confirmation for large experiments
        if exp_info['runs'] > 10:
            if not Confirm.ask(f"[red]This experiment has {exp_info['runs']} runs. Really delete?[/red]"):
                self.console.print("[yellow]Deletion cancelled.[/yellow]")
                return
        
        try:
            # Create backup option
            if Confirm.ask("Create a backup before deleting?"):
                backup_name = f"{exp_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_path = exp_path.parent / backup_name
                shutil.copytree(exp_path, backup_path)
                self.console.print(f"[green]Backup created at: {backup_path}[/green]")
            
            # Delete the experiment
            shutil.rmtree(exp_path)
            self.console.print(f"[green]✅ Successfully deleted experiment '{exp_name}'[/green]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Error deleting experiment: {e}[/red]")

    def export_results(self, exp_info: Dict):
        """Export experiment results to different formats."""
        exp_dir = Path(exp_info['path'])
        exp_name = exp_info['name']
        
        self.console.print(f"\n[bold blue]📤 Export Results: {exp_name}[/bold blue]")
        
        # Check available data
        csv_file = exp_dir / 'run_table.csv'
        has_csv = csv_file.exists()
        
        if not has_csv:
            self.console.print("[red]No results to export (no run_table.csv found)[/red]")
            return
            
        export_options = [
            "1. Copy CSV file to current directory",
            "2. Create summary report (JSON)",
            "3. Export to Excel format", 
            "4. Create ZIP archive with all results",
            "0. Cancel"
        ]
        
        for option in export_options:
            self.console.print(option)
            
        choice = Prompt.ask("Choose export option", choices=['0', '1', '2', '3', '4'])
        
        if choice == '0':
            return
            
        try:
            if choice == '1':
                # Copy CSV
                dest = Path(f"{exp_name}_results.csv")
                shutil.copy2(csv_file, dest)
                self.console.print(f"[green]✅ CSV exported to: {dest}[/green]")
                
            elif choice == '2':
                # Create summary
                summary = self.create_summary_report(exp_dir)
                dest = Path(f"{exp_name}_summary.json")
                with open(dest, 'w') as f:
                    json.dump(summary, f, indent=2)
                self.console.print(f"[green]✅ Summary exported to: {dest}[/green]")
                
            elif choice == '3':
                self.console.print("[yellow]Excel export requires pandas and openpyxl. Install with: pip install pandas openpyxl[/yellow]")
                
            elif choice == '4':
                # Create ZIP archive
                dest = Path(f"{exp_name}_complete.zip")
                shutil.make_archive(dest.stem, 'zip', exp_dir)
                self.console.print(f"[green]✅ Archive created: {dest}[/green]")
                
        except Exception as e:
            self.console.print(f"[red]❌ Export failed: {e}[/red]")

    def create_summary_report(self, exp_dir: Path) -> Dict:
        """Create a summary report of experiment results."""
        summary = {
            'experiment_name': exp_dir.name,
            'generated_at': datetime.now().isoformat(),
            'summary': {}
        }
        
        # Analyze CSV if available
        csv_file = exp_dir / 'run_table.csv'
        if csv_file.exists():
            try:
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    
                summary['summary'] = {
                    'total_runs': len(rows),
                    'completed_runs': sum(1 for row in rows if row.get('__done', '').upper() == 'DONE'),
                    'columns': list(rows[0].keys()) if rows else [],
                    'first_run': rows[0] if rows else None,
                    'run_sample': rows[:3] if len(rows) > 3 else rows
                }
                
            except Exception as e:
                summary['csv_error'] = str(e)
        
        return summary

    def run(self):
        """Main application loop."""
        self.show_header()
        
        while True:
            # Scan for experiments
            experiments = self.scan_experiments()
            
            if not experiments:
                self.console.print("[yellow]No experiments found in the experiments directory.[/yellow]")
                self.console.print("Make sure you're running this from the llm-experiment-runner directory.")
                break
                
            # Show experiments table
            self.show_experiments_table(experiments)
            
            self.console.print("\n[bold]Options:[/bold]")
            self.console.print("• Enter experiment ID to view details")
            self.console.print("• 'c<ID>' to view CSV content (e.g., 'c1')")
            self.console.print("• 'd<ID>' to delete (e.g., 'd1')")
            self.console.print("• 'e<ID>' to export (e.g., 'e1')")
            self.console.print("• 'dall' to delete all experiments")
            self.console.print("• 'eall' to export all experiments")
            self.console.print("• 'r' to refresh")
            self.console.print("• 'q' to quit")
            
            choice = Prompt.ask("\nChoice").strip().lower()
            
            if choice == 'q':
                break
            elif choice == 'r':
                continue
            elif choice == 'dall':
                self.delete_all_experiments(experiments)
            elif choice == 'eall':
                self.export_all_experiments(experiments)
            elif choice.startswith('c') and len(choice) > 1:
                try:
                    exp_id = int(choice[1:]) - 1
                    if 0 <= exp_id < len(experiments):
                        exp_info = experiments[exp_id]
                        csv_file = Path(exp_info['path']) / 'run_table.csv'
                        if csv_file.exists():
                            self.view_csv_content(csv_file)
                        else:
                            self.console.print(f"[red]No CSV file found for experiment '{exp_info['name']}'[/red]")
                            Prompt.ask("Press Enter to continue")
                    else:
                        self.console.print("[red]Invalid experiment ID[/red]")
                except ValueError:
                    self.console.print("[red]Invalid format. Use 'c1', 'c2', etc.[/red]")
            elif choice.startswith('d') and len(choice) > 1:
                try:
                    exp_id = int(choice[1:]) - 1
                    if 0 <= exp_id < len(experiments):
                        self.delete_experiment(experiments[exp_id])
                    else:
                        self.console.print("[red]Invalid experiment ID[/red]")
                except ValueError:
                    self.console.print("[red]Invalid format. Use 'd1', 'd2', etc.[/red]")
            elif choice.startswith('e') and len(choice) > 1:
                try:
                    exp_id = int(choice[1:]) - 1
                    if 0 <= exp_id < len(experiments):
                        self.export_results(experiments[exp_id])
                    else:
                        self.console.print("[red]Invalid experiment ID[/red]")
                except ValueError:
                    self.console.print("[red]Invalid format. Use 'e1', 'e2', etc.[/red]")
            else:
                try:
                    exp_id = int(choice) - 1
                    if 0 <= exp_id < len(experiments):
                        self.view_experiment_details(experiments[exp_id])
                        Prompt.ask("\nPress Enter to continue")
                    else:
                        self.console.print("[red]Invalid experiment ID[/red]")
                except ValueError:
                    self.console.print("[red]Please enter a valid number or command[/red]")
            
            self.console.clear()
        
        self.console.print("\n[bold blue]👋 Thank you for using Results Explorer![/bold blue]")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
LLM Experiment Results Explorer

Usage: python results_explorer.py

Features:
- View all experiment results
- Browse CSV content with advanced filtering and search
- Delete unwanted experiments (with backup option)
- Export results to various formats
- Browse experiment details and run data
- Bulk operations for managing multiple experiments

Commands:
- Enter number: View experiment details
- c<ID>: View CSV content (e.g., c1)
- d<ID>: Delete experiment (e.g., d1)  
- e<ID>: Export results (e.g., e1)
- dall: Delete all experiments (with backup)
- eall: Export all experiments (multiple formats)
- r: Refresh list
- q: Quit

Make sure to run this from the llm-experiment-runner directory.
        """)
        return
        
    try:
        explorer = ResultsExplorer()
        explorer.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()