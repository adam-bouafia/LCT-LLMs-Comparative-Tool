from ...ProgressManager.RunTable.Models.RunProgress import RunProgress
from ...ConfigValidator.CustomErrors.ExperimentOutputErrors import ExperimentOutputFileDoesNotExistError
from ...ProgressManager.Output.OutputProcedure import OutputProcedure as output
from ...ProgressManager.Output.BaseOutputManager import BaseOutputManager

from tempfile import NamedTemporaryFile
import shutil
import csv
import os
from typing import Dict, List


class CSVOutputManager(BaseOutputManager):
    def read_run_table(self) -> List[Dict]:
        read_run_table = []
        try:
            with open(self._experiment_path / 'run_table.csv', 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # if value was integer, stored as string by CSV writer, then convert back to integer.
                    for key, value in row.items():
                        if value.isnumeric():
                            row[key] = int(value)

                        if key == '__done':
                            row[key] = RunProgress[value]

                    read_run_table.append(row)
            
            return read_run_table
        except:
            raise ExperimentOutputFileDoesNotExistError

    def write_run_table(self, run_table: List[Dict]):
        try:
            with open(self._experiment_path / 'run_table.csv', 'w', newline='') as myfile:
                writer = csv.DictWriter(myfile, fieldnames=list(run_table[0].keys()))
                writer.writeheader()
                for data in run_table:
                    data['__done'] = data['__done'].name
                    writer.writerow(data)
        except:
            raise ExperimentOutputFileDoesNotExistError

    # TODO: Nice To have
    def shuffle_experiment_run_table(self):
        pass
    
    def update_row_data(self, updated_row: dict):
        csv_path = self._experiment_path / 'run_table.csv'
        
        # If CSV doesn't exist, create it first
        if not csv_path.exists():
            output.console_log(f"CSV file not found at {csv_path}, creating new one...")
            try:
                # Try to read existing data first
                existing_data = self.read_run_table()
                self.write_run_table(existing_data + [updated_row])
                output.console_log(f"CSVManager: Created new CSV and added row {updated_row['__run_id']}")
                return
            except:
                # If read fails, create a minimal CSV with just this row
                updated_row['__done'] = updated_row['__done'].name if hasattr(updated_row['__done'], 'name') else updated_row['__done']
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=list(updated_row.keys()))
                    writer.writeheader()
                    writer.writerow(updated_row)
                output.console_log(f"CSVManager: Created minimal CSV with row {updated_row['__run_id']}")
                return
        
        # Normal update process for existing CSV
        tempfile = NamedTemporaryFile(mode='w', delete=False)

        try:
            with open(csv_path, 'r') as csvfile, tempfile:
                reader = csv.DictReader(csvfile, fieldnames=list(updated_row.keys()))
                writer = csv.DictWriter(tempfile, fieldnames=list(updated_row.keys()))

                for row in reader:
                    if row['__run_id'] == updated_row['__run_id']:
                        # When the row is updated, it is an ENUM value again.
                        # Write as human-readable: enum_value.name
                        updated_row['__done'] = updated_row['__done'].name
                        writer.writerow(updated_row)
                    else:
                        writer.writerow(row)

            shutil.move(tempfile.name, csv_path)
            output.console_log(f"CSVManager: Updated row {updated_row['__run_id']}")
        except Exception as e:
            output.console_log(f"CSVManager: Error updating row - {str(e)}")
            # Clean up temp file if something went wrong
            try:
                os.unlink(tempfile.name)
            except:
                pass

        # with open(self.experiment_path + '/run_table.csv', 'w', newline='') as myfile:
        #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        #     wr.writerow(updated_row)

    # for row in reader:
    # if name == row['name']:
    #     row['name'] = input("enter new name for {}".format(name))
    # # write the row either way
    # writer.writerow({'name': row['name'], 'number': row['number'], 'address': row['address']})