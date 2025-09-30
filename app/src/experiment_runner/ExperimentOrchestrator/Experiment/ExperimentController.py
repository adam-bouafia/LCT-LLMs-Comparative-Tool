import time
import multiprocessing

from ...ConfigValidator.Config.Models.Metadata import Metadata
from ...ConfigValidator.CustomErrors.BaseError import BaseError
from ...ProgressManager.Output.JSONOutputManager import JSONOutputManager
from ...ProgressManager.RunTable.Models.RunProgress import RunProgress
from ...ConfigValidator.Config.Models.OperationType import OperationType
from ...EventManager.Models.RunnerEvents import RunnerEvents
from ...ProgressManager.Output.CSVOutputManager import CSVOutputManager
from ...ExperimentOrchestrator.Experiment.Run.RunController import RunController
from ...ConfigValidator.Config.RunnerConfig import RunnerConfig
from ...ProgressManager.Output.OutputProcedure import OutputProcedure as output
from ...EventManager.EventSubscriptionController import EventSubscriptionController
from ...ConfigValidator.CustomErrors.ProgressErrors import AllRunsCompletedOnRestartError


###     =========================================================
###     |                                                       |
###     |                  ExperimentController                 |
###     |       - Init and perform runs of correct type         |
###     |       - Perform experiment overhead                   |
###     |       - Perform run overhead (time_btwn_runs)         |
###     |       - Signal experiment end (ClientRunner)          |
###     |                                                       |
###     |       * Experiment config that should be used         |
###     |         throughout the program is declared here       |
###     |         and should not be redeclared (only passed)    |
###     |                                                       |
###     =========================================================
class ExperimentController:

    def __init__(self, config: RunnerConfig, metadata: Metadata):
        self.config = config
        self.metadata = metadata

        self.csv_data_manager = CSVOutputManager(self.config.experiment_path)
        self.json_data_manager = JSONOutputManager(self.config.experiment_path)
        self.run_table = self.config.create_run_table_model().generate_experiment_run_table()

        # Create experiment output folder, automatically rename existing ones with timestamp
        self.restarted = False
        try:
            self.config.experiment_path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            # Automatically rename existing folder with timestamp to avoid conflicts
            from datetime import datetime
            import shutil
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.config.experiment_path.parent / f"{self.config.experiment_path.name}_backup_{timestamp}"
            
            output.console_log(f"Experiment folder exists. Renaming to: {backup_path.name}")
            shutil.move(str(self.config.experiment_path), str(backup_path))
            
            # Now create fresh experiment folder
            self.config.experiment_path.mkdir(parents=True, exist_ok=True)
            self.restarted = False
            return

        # Write initial run table and metadata for fresh experiment
        self.csv_data_manager.write_run_table(self.run_table)
        self.json_data_manager.write_metadata(self.metadata)

        output.console_log("Experiment run table created...")

    def do_experiment(self):
        output.console_log("Experiment setup completed...")

        # -- Before experiment
        # TODO: From a user perspective, it would be nice to know if this is a restarted experiment or not (in case something failed)
        output.console_log("Calling before_experiment config hook")
        EventSubscriptionController.raise_event(RunnerEvents.BEFORE_EXPERIMENT)

        # -- Experiment
        for variation in self.run_table:
            if variation['__done'] == RunProgress.DONE:
                continue

            output.console_log("Calling before_run config hook")
            EventSubscriptionController.raise_event(RunnerEvents.BEFORE_RUN)

            run_controller = RunController(variation, self.config, (self.run_table.index(variation) + 1), len(self.run_table))
            perform_run = multiprocessing.Process(
                target=run_controller.do_run,
                args=[]
            )
            perform_run.start()
            perform_run.join()

            time_btwn_runs = self.config.time_between_runs_in_ms
            if time_btwn_runs > 0:
                output.console_log(f"Run fully ended, waiting for: {time_btwn_runs}ms == {time_btwn_runs / 1000}s")
                time.sleep(time_btwn_runs / 1000)

            if self.config.operation_type is OperationType.SEMI:
                EventSubscriptionController.raise_event(RunnerEvents.CONTINUE)

        output.console_log("Experiment completed...")

        # -- After experiment
        output.console_log("Calling after_experiment config hook")
        EventSubscriptionController.raise_event(RunnerEvents.AFTER_EXPERIMENT)
