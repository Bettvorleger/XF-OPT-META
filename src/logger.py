import timeit
import json
from pathlib import Path
from re import findall
from functools import partial
from datetime import datetime
from config import logger_run_params
import pandas as pd
import numpy as np


class Logger:
    # specify run mode constants
    MODE_RUN = "run"
    MODE_EXPERIMENT = "exp"
    MODE_OPTIMIZE = "opt"

    def __init__(self, path="output/", mode=MODE_RUN, suffix_number=0, algorithm='hsppbo') -> None:
        """
        Create a logger instance for creating all relevant files for a given algorithm mode
        Modes are:
            - MODE_RUN(run):        normal execution of one run of the algorithm, creating a run log
            - MODE_EXPERIMENT(exp): creating a run log for every run and averaged results for all runs
            - MDOE_OPTIMIZE(opt):   creating a log of every optimizer run (raw output including models built and all parameters used)
                                    and a summary best parameter combinations for each run

        Args:
            path (str, optional):           Path to the output folder relative to the script. Defaults to "output/".
            mode (_type_, optional):        Setting the mode of the logger, changing its behaviour and created files. Defaults to MODE_RUN.
            suffix_number (int, optional):  Folder suffix (e.g. run_1 for suffix 1) for the created folder. Can be manually set if needed. 
                                            Defaults to 0 for auto-detection of next greater number within the folder.
        """
        self.path = path
        self.mode = mode
        self.path_prefix = {self.MODE_RUN: "run_",
                            self.MODE_EXPERIMENT: "exp_", self.MODE_OPTIMIZE: "opt_"}
        self.suffix_number = suffix_number
        if (suffix_number == 0):
            self.suffix_number = self.init_folder()

        # params that are logged each algorithm run
        self.run_params = logger_run_params[algorithm]

        self.info = {}

        # initializing the mode with the corresponding method
        modes = {
            self.MODE_RUN: self.init_run,
            self.MODE_EXPERIMENT: self.init_experiment,
            self.MODE_OPTIMIZE: self.init_optimize
        }
        self.init_mode = partial(modes[self.mode])

    def init_run(self) -> None:
        """
        Init the run logger mode
        """
        self.create_info_log()
        self.run_io = self.create_file_wrapper("run.csv")
        self.create_run_log_header()
        self.starttime = timeit.default_timer()
        self.run_list = []

    def init_experiment(self, runs: int) -> None:
        """
        Init the experiment logger mode

        Args:
            runs (int): Max number of algorithm runs performed
        """
        self.max_runs = runs
        self.create_info_log()
        self.run_io = self.create_file_wrapper("exp_run_1.csv")
        self.create_run_log_header()
        self.starttime = timeit.default_timer()
        self.run_list = []
        self.exp_run_list = []

    def init_optimize(self, params: list[tuple], opt_algo: str) -> None:
        """
        Init the optimize logger mode

        Args:
            params (list[tuple[str, skopt.Dimension]]): List of tuples, specifying the parameters to optimize, described by the string param name and the dimension of the param
            opt_algo (str): Name of optimization algorithm used
        """
        self.params = params
        self.info["optimizer"] = opt_algo
        self.create_info_log()
        self.best_params_list = []

    def init_folder(self) -> int:
        """
        Initialize the folder and file for logging.
        Finds the current next run number within the given folder,
        starting with 1 of the folder is new or counting up until the highes number is found.

        Returns:
            int: The number of the current run
        """
        suffix_number = 0
        Path(self.path).mkdir(parents=True, exist_ok=True)

        for p in Path(self.path).iterdir():
            if p.is_dir:
                num = findall(self.path_prefix[self.mode]+"(.*)", p.name)
                if not num:
                    continue
                num = int(num[0])
                suffix_number = num if num > suffix_number else suffix_number
        suffix_number += 1
        Path("".join((self.path, self.path_prefix[self.mode],
             str(suffix_number), "/"))).mkdir(parents=True, exist_ok=True)

        return suffix_number

    def create_file_wrapper(self, filename: str) -> None:
        """
        Initializes a file wrapper for logging

        Args:
            filename (str): Name of the file that should be created

        Returns:
            TextIOWrapper: Wrapper for the opened logging file
        """
        return open("".join((self.path, self.path_prefix[self.mode], str(self.suffix_number), "/", filename)), "a")

    def set_info(self, info: dict):
        """
        Set the info about the current runtime environment and used parameters for the modules

        Args:
            info (dict): info about all the algorithm components and parameters
        """
        self.info = info

    def create_info_log(self):
        """
        Create log about the current runtime environment and used parameters for the modules 

        Args:
            info (dict): info about all the algorithm components and parameters
        """
        io_file = self.create_file_wrapper("info.json")
        self.info['datetime'] = datetime.now()
        io_file.write(json.dumps(self.info, indent=4, default=str))
        io_file.close()

    def create_opt_files(self, opt_results, run: int) -> None:
        """
        Create the log of the raw optimizer results (opt_log) and the parameters used for each optimizer iteration.
        Created for each run

        Args:
            opt_results (_type_): _description_
            run (int, optional): _description_. Defaults to 1.
        """
        # dump out all the data from the optimizer run
        io_file = self.create_file_wrapper("opt_log_"+str(run)+".json")
        io_file.write(json.dumps(opt_results, indent=4, default=str))
        io_file.close()

        io_file = self.create_file_wrapper("opt_run_"+str(run)+".csv")
        # write the opt run header
        io_file.write("iteration;")
        for p in self.params:
            io_file.write("".join((p[0], ";")))
        io_file.write("func_val\n")

        # write the opt run data
        for i, param_vals in enumerate(opt_results.x_iters):
            io_file.write("".join((str(i+1), ";")))
            for x in param_vals:
                io_file.write("".join((str(x), ";")))
            io_file.write("".join((str(opt_results.func_vals[i]), "\n")))
        io_file.close()

        # write opt run best params to dict
        self.best_params_list.append([*opt_results.x, opt_results.fun])

    def create_opt_best_params(self) -> None:
        """
        Create the summary of the best param configurations found during each optimizer run
        """
        io_file = self.create_file_wrapper("opt_best_params.csv")
        io_file.write("run;")
        for p in self.params:
            io_file.write("".join((p[0], ";")))
        io_file.write("func_val\n")

        for i, param_vals in enumerate(self.best_params_list):
            io_file.write(str(i+1)+";")
            io_file.write(";".join(str(x) for x in param_vals))
            io_file.write("\n")
        io_file.close()

    def add_exp_run(self, run: int) -> None:
        if run == 1:
            self.exp_run_list = [
                [[] for j in range(len(self.run_list))] for i in range(len(self.run_params))]
        #run_list_reversed = (pd.DataFrame(self.run_list).T).values.tolist()
        for i_key, i_val in enumerate(self.run_list):
            for p_key in range(len(self.run_params)):
                self.exp_run_list[p_key][i_key].append(i_val[p_key])

        # prevents from adding an empty file at the end
        if run < self.max_runs:
            self.run_io = self.create_file_wrapper(
                "exp_run_"+str(run+1)+".csv")
            self.create_run_log_header()
            self.starttime = timeit.default_timer()
            self.run_list.clear()

    def create_exp_avg_run(self) -> None:
        io_file = self.create_file_wrapper("avg_run.csv")
        df = pd.DataFrame(self.exp_run_list, index=(self.run_params))
        #df.iloc[[0]] = df.iloc[[0]].apply(lambda x: x[0])
        #print(df.iloc[[0]])
        df.to_csv(io_file)
        io_file.close()

    def create_run_log_header(self) -> None:
        """
        Create the header for the csv file of the run log
        """
        self.run_io.write(";".join(self.run_params) + "\n")

    def log_iteration(self, it_num: int, func_evals: int, swap_num: int, reaction: bool, best_solution: float) -> None:
        """
        Log the current iteration of the running algorithm

        TODO: implement hanndling for different algorithms/parameters

        Args:
            it_num (int): Number of the current iteration
            func_evals (int): Number of function evaluations needes for calculating the solution
            swap_num (int): Number of swaps that happended during that iteration
            reaction (bool): If the change handling reaction was triggered or not 
            best_solution (float): The iteration best solution quality
        """
        if self.mode == self.MODE_OPTIMIZE:
            return

        runtime = timeit.default_timer() - self.starttime
        self.run_list.append(
            [it_num, runtime, func_evals, swap_num, reaction, best_solution])

        self.run_io.write("%d;%0.3f;%d;%d;%r;%f\n" %
                          (it_num, runtime, func_evals, swap_num, reaction, best_solution))

    def close_run_logger(self) -> None:
        """
        Close the run logger io file
        """
        self.run_io.close()
