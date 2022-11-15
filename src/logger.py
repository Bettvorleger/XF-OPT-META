import timeit
from pathlib import Path
from re import findall


class Logger():
    def __init__(self, path="output/", mode="run", suffix_number=0) -> None:
        self.path = path
        self.mode = mode
        self.path_prefix = {"run": "run_", "exp": "exp_", "opt": "opt_"}
        self.suffix_number = suffix_number
        if (suffix_number == 0):
            self.suffix_number = self.init_folder()
        self.log_list = []
        self.run_io = self.create_file_wrapper("run.csv")
        self.create_run_log_header()
        self.starttime = timeit.default_timer()

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

    def create_info_log(self, params):
        io_file = self.create_file_wrapper("info.log")

        for k, v in params.items():
            if k in ("logger", "rng"):
                continue
            io_file.write(k+": "+str(v)+"\n")

    def create_run_log_header(self):
        self.run_io.write(
            "iteration;abs_runtime;func_evals;swaps,reaction;best_solution\n")

    def log_iteration(self, it_num: int, func_evals: int, swap_num: int, reaction: bool, best_solution: float):
        """
        Log the current iteration of the running algorithm

        Args:
            it_num (int): Number of the current iteration
            swap_num (int): Number of swaps that happended during that iteration
            reaction (bool): If the change handling reaction was triggered or not 
            best_solution (float): The iteration best solution quality
        """
        runtime = timeit.default_timer() - self.starttime
        self.log_list.append(
            [it_num, runtime, func_evals, swap_num, reaction, best_solution])
        self.run_io.write("%d;%0.3f;%d;%d;%r;%f\n" %
                          (it_num, runtime, func_evals, swap_num, reaction, best_solution))

    def create_run_results(self, optimal_solution):
        io_file = self.create_file_wrapper("results.csv")
        #rpd_list = [((x[4] - optimal_solution) / optimal_solution) for x in self.log_list]
        #time_list =  [x[1] for x in self.log_list]
