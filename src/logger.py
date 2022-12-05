import timeit
import json
from pathlib import Path
from re import findall
from functools import partial
from datetime import datetime
import numpy as np


class Logger():
    MODE_RUN = "run"
    MODE_EXPERIMENT = "exp"
    MODE_OPTIMIZE = "opt"

    def __init__(self, path="output/", mode=MODE_RUN, suffix_number=0) -> None:
        self.path = path
        self.mode = mode
        modes = {
            self.MODE_RUN: self.run,
            self.MODE_EXPERIMENT: self.experiment,
            self.MODE_OPTIMIZE: self.optimize
        }
        self.path_prefix = {self.MODE_RUN: "run_",
                            self.MODE_EXPERIMENT: "exp_", self.MODE_OPTIMIZE: "opt_"}
        self.suffix_number = suffix_number
        if (suffix_number == 0):
            self.suffix_number = self.init_folder()

        self.init_mode = partial(modes[self.mode])

        self.run_params = ["iteration", "abs_runtime",
                           "func_evals", "swaps", "reaction", "best_solution"]

    def run(self):
        self.run_io = self.create_file_wrapper("run.csv")
        self.create_run_log_header()
        self.starttime = timeit.default_timer()
        self.run_list = []

    def experiment(self, runs):
        self.starttime = timeit.default_timer()
        self.run_list = []
        self.exp_run_list = [
            [[] for j in range(runs)] for i in range(len(self.run_params))]

    def optimize(self, params):
        self.params = params
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

    def create_file_wrapper(self, filename: str):
        """
        Initializes a file wrapper for logging

        Args:
            filename (str): Name of the file that should be created

        Returns:
            TextIOWrapper: Wrapper for the opened logging file
        """
        return open("".join((self.path, self.path_prefix[self.mode], str(self.suffix_number), "/", filename)), "a")

    def create_info_log(self, info: dict):
        io_file = self.create_file_wrapper("info.json")
        info['datetime'] = datetime.now()
        io_file.write(json.dumps(info, indent=4, default=str))
        io_file.close()

    def create_opt_files(self, opt_results, run=1):
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

    def create_opt_best_params(self):
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

    def add_run_exp(self):
        for i_key, i_val in enumerate(self.run_list.copy()):
            for k in range(0, len(self.exp_run_list)):
                self.exp_run_list[k][i_key].append(i_val[k])

        self.run_list.clear()
        self.starttime = timeit.default_timer()

    def create_exp_avg_run(self):
        io_file = self.create_file_wrapper("avg_run.csv")
        io_file.write(";".join(self.run_params) + "\n")

        io_file.close()

    def create_run_log_header(self):
        self.run_io.write(";".join(self.run_params) + "\n")

    def log_iteration(self, it_num: int, func_evals: int, swap_num: int, reaction: bool, best_solution: float):
        """
        Log the current iteration of the running algorithm

        Args:
            it_num (int): Number of the current iteration
            swap_num (int): Number of swaps that happended during that iteration
            reaction (bool): If the change handling reaction was triggered or not 
            best_solution (float): The iteration best solution quality
        """

        if self.mode == self.MODE_EXPERIMENT or self.mode == self.MODE_RUN:
            runtime = timeit.default_timer() - self.starttime
            self.run_list.append(
                [it_num, runtime, func_evals, swap_num, reaction, best_solution])

        if self.mode != self.MODE_RUN:
            return

        self.run_io.write("%d;%0.3f;%d;%d;%r;%f\n" %
                          (it_num, runtime, func_evals, swap_num, reaction, best_solution))

    def create_run_results(self, optimal_solution):
        io_file = self.create_file_wrapper("results.csv")
        #rpd_list = [((x[4] - optimal_solution) / optimal_solution) for x in self.log_list]
        #time_list =  [x[1] for x in self.log_list]
