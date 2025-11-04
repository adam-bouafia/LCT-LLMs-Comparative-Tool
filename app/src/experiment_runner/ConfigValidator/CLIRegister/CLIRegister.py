import os
import uuid
import inspect
import subprocess
import sys
from typing import List
from shutil import copyfile
from tabulate import tabulate

from ...ConfigValidator.Config.RunnerConfig import RunnerConfig
from ...ExperimentOrchestrator.Misc.BashHeaders import BashHeaders
from ...ExperimentOrchestrator.Misc.PathValidation import is_path_exists_or_creatable_portable
from ...ProgressManager.Output.OutputProcedure import OutputProcedure as output
from ...ConfigValidator.CustomErrors.CLIErrors import *

class ConfigCreate:
    @staticmethod
    def description_params() -> str:
        return "[path_to_user_specified_dir]"

    @staticmethod
    def description_short() -> str:
        return "Creates a config file in the `experiments', or user-specified, directory"

    @staticmethod
    def description_long() -> str:
        output.console_log_bold("... TODO give usage instructons ...")

    @staticmethod
    def execute(args=None) -> None:
        try:
            destination = ""
            if args is None:
                filepath = __file__.split('/')
                filepath.pop()
                filepath = '/'.join(filepath) + "/../../../examples/"
                destination = os.path.abspath(filepath)
            else:
                if len(args) == 3:
                    destination = args[2]
                else:
                    raise CommandNotRecognisedError
        except:
            raise CommandNotRecognisedError

        if destination[-1] != "/":
            destination += "/"

        if not is_path_exists_or_creatable_portable(destination):
            raise InvalidUserSpecifiedPathError(destination)
        
        module = RunnerConfig
        src = inspect.getmodule(module).__file__
        dest_folder = destination
        config_unique_name = 'RunnerConfig-' + str(uuid.uuid1()) + '.py'
        destination += config_unique_name
        copyfile(src, destination)
        output.console_log_OK(
            f"Successfully created new config with unique identifier in: {dest_folder}" +
            f"\nWith the unique name (please rename): {config_unique_name}"
        )

class Prepare:
    @staticmethod
    def description_params() -> str:
        return ""

    @staticmethod
    def description_short() -> str:
        return "(WIP) Prepare system -- Install all dependencies"

    @staticmethod
    def description_long() -> str:
        output.console_log_bold("Prepare will install all the user's required dependencies to be able to run experiment-runner on their system.")

    @staticmethod
    def execute(args=None) -> None:
        pass

class ResultsExplorer:
    @staticmethod
    def description_params() -> str:
        return ""

    @staticmethod
    def description_short() -> str:
        return "Launch the Results Explorer - view, analyze, and manage experiment results"

    @staticmethod
    def description_long() -> str:
        output.console_log_bold("Results Explorer")
        print("Launch an interactive tool to:")
        print("• Browse all experiments")
        print("• View CSV content with statistics and filtering")
        print("• Delete individual experiments or bulk delete all")
        print("• Export experiments individually or bulk export all")
        print("• Manage experiment lifecycle")
        print("\nThe Results Explorer provides a user-friendly interface for comprehensive")
        print("experiment management and analysis.")

    @staticmethod
    def execute(args=None) -> None:
        # Find the results_explorer.py script
        script_path = None
        
        # Try different possible locations
        possible_paths = [
            "../llm-experiment-runner/results_explorer.py",
            "../../llm-experiment-runner/results_explorer.py", 
            "../../../llm-experiment-runner/results_explorer.py",
            "results_explorer.py"
        ]
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        for path in possible_paths:
            full_path = os.path.abspath(os.path.join(current_dir, path))
            if os.path.exists(full_path):
                script_path = full_path
                break
        
        if script_path is None:
            output.console_log_FAIL("❌ Results Explorer not found!")
            output.console_log("Please ensure results_explorer.py is in the llm-experiment-runner directory")
            return
            
        try:
            output.console_log("🔍 Launching Results Explorer...")
            # Run the results explorer script
            os.chdir(os.path.dirname(script_path))
            subprocess.run([sys.executable, "results_explorer.py"])
        except KeyboardInterrupt:
            output.console_log("\n👋 Results Explorer closed")
        except Exception as e:
            output.console_log_FAIL(f"❌ Error launching Results Explorer: {str(e)}")

class Help:
    @staticmethod
    def description_params() -> str:
        return ""

    @staticmethod
    def description_short() -> str:
        return "An overview of all available commands"

    @staticmethod
    def description_long() -> str:
        print(BashHeaders.BOLD + "--- EXPERIMENT_RUNNER HELP ---" + BashHeaders.ENDC)
        print("\n%-*s  %s" % (10, "Usage:", "python experiment-runner/ <path_to_config.py>"))
        print("%-*s  %s" % (10, "Utility:", "python experiment-runner/ <command>"))

        print("\nAvailable commands:\n")
        print(tabulate([(k, v.description_params()) for k, v in CLIRegister.register.items()], ["Command", "Parameters"]))

        print("\nHelp can be called for each command:")
        print(BashHeaders.WARNING + "example: " + BashHeaders.ENDC + "python experiment-runner/ prepare help")

    @staticmethod
    def execute(args=None) -> None:
        Help.description_long()

class CLIRegister:
    register = {
        "config-create":    ConfigCreate,
        "prepare":          Prepare,
        "results-explorer": ResultsExplorer,
        "help":             Help
    }

    @staticmethod 
    def parse_command(args: List):
        try:
            command_class = CLIRegister.register.get(args[1])
        except:
            raise CommandNotRecognisedError

        if len(args) == 2:
            command_class.execute()
        elif len(args) > 2:
            if args[2] == 'help':
                command_class.description_long()
            else:
                command_class.execute(args)