from typing import Generator
from scipy.optimize import OptimizeResult
from problem import Problem
from tsp import TSP
from config import output_folder_prefix, params
import json
from pathlib import Path
import dill as pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skopt.space import Space, Integer, Real
from skopt.plots import plot_convergence, plot_objective
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None


class Analyzer:
    # specify mode constants
    MODE_RUN = "run"
    MODE_EXPERIMENT = "exp"
    MODE_OPTIMIZE = "opt"

    def __init__(self, mode=MODE_RUN, obj_algorithm='hsppbo', results_path="output/") -> None:
        self.mode = mode
        self.path_prefix = output_folder_prefix[mode]
        self.results_path = results_path
        self.obj_algorithm = obj_algorithm

    def load_result_folder(self, result_num) -> Generator[Path, None, None]:
        path = "".join(
            [self.results_path, self.path_prefix, str(result_num), '/'])
        if Path(path).exists():
            return Path(path).iterdir()
        return None

    def create_run_results(self, optimal_solution):
        io_file = self.create_file_wrapper("results.csv")
        # rpd_list = [((x[4] - optimal_solution) / optimal_solution) for x in self.log_list]
        # time_list =  [x[1] for x in self.log_list]

    def create_convergence_plot(self, result_nums: list[int]):

        results = []
        opt_solution = None
        opt = ""
        for n in result_nums:
            res = []

            # load IO wrapper for accesing files from run
            files = self.load_result_folder(n)
            if files:
                for file in files:
                    if file.name == 'info.json':
                        opt = get_optimizer_type(file.as_posix())
                        opt_solution = get_optimal_solution(file.as_posix())
                    elif 'opt_log' and 'json' in file.name:
                        with open(file.as_posix(), 'r') as f:
                            log = json.load(
                                f, object_hook=lambda d: OptimizeResult(**d))
                            f.close()
                            if opt_solution:
                                log.func_vals = np.array(
                                    [(x-opt_solution)/opt_solution for x in log.func_vals])
                            else:
                                log.func_vals = np.array(log.func_vals)
                        res.append(log)
                results.append((opt, res))

        if results:
            plot = plot_convergence(*results)
            if opt_solution:
                plot.set_ylim([-0.02, 0.25])
            plot.legend(loc="best", prop={'size': 8}, numpoints=1)
            plt.show()

    def create_partial_dep_plot(self, result_num: int, n_points=40):
        results = []
        problem = ""
        opt_solution = None
        opt = ""

        # load IO wrapper for accesing files from run
        files = self.load_result_folder(result_num)
        if files:
            for file in files:
                if file.name == 'info.json':
                    problem = get_problem_name(file.as_posix())
                    opt = get_optimizer_type(file.as_posix())
                    opt_solution = get_optimal_solution(file.as_posix())
                elif 'opt_log' and 'pkl' in file.name:
                    with open(file.as_posix(), 'rb') as f:
                        log = pickle.load(f)
                        f.close()
                    results.append(log)

        dimensions = [d[0] for d in params[self.mode][self.obj_algorithm]]
        for i, res in enumerate(results):
            plot_objective(res, dimensions=dimensions, n_points=n_points)
            plt.suptitle('Partial Dependence (opt_%d, %s, %s, run %d of %d)' %
                         (result_num, opt, problem, i, len(results)))
        plt.show()


def get_optimizer_type(path: str) -> str:
    with open(path, 'r') as f:
        info = json.load(f)
        f.close()
    return info['optimizer']


def get_problem_name(path: str) -> str:
    with open(path, 'r') as f:
        info = json.load(f)
        f.close()
    return info['problem']['name']


def get_optimal_solution(path: str) -> int:
    problem = get_problem_name(path)
    p = TSP(problem, load_instance=False)
    return p.get_optimal_solution()


def create_problem_metadata(problem: Problem, metadata_filepath='../problems/metadata.json'):
    if Path(metadata_filepath).exists():
        with open(metadata_filepath, 'r') as f:
            metadata = json.load(f)
            f.close()
    else:
        metadata = {'tsp': {'stsp': {}}}

    if problem.TYPE == 'TSP':
        instance = metadata['tsp']['stsp'][problem.instance.name] = {}
    else:
        raise ValueError('Problem type {problem.TYPE} not supported')

    instance['mean'] = problem.get_mean_distance()
    instance['median'] = problem.get_median_distance()
    instance['coeff_var'] = problem.get_coefficient_of_variation()
    instance['qdc'] = problem.get_quartile_dispersion_coefficient()
    instance['R'] = problem.get_regularity_index()
    instance['eigen1'] = problem.get_first_eigenvalue()

    with open(metadata_filepath, "w") as outfile:
        json.dump(metadata, outfile, indent=4, default=str)


def create_problem_cluster(metadata_filepath='../problems/metadata.json', output_filepath='../problems/', plot_cluster=False, n_clusters_kmeans=5):

    with open(metadata_filepath, 'r') as f:
        metadata = json.load(f)
        f.close()

    instances = {}
    clusters_1 = {
        'heavy_cluster': [],
        'semi_cluster': [],
        'random': [],
        'semi_regular': [],
        'regular': [],
    }

    # Method 1: Using value areas for the regularity index R to distinct between structure
    for i, v in metadata['tsp']['stsp'].items():
        if v['mean'] is None:
            continue

        instances[i] = [*v.values()]

        if v['R'] < 0.3:
            clusters_1['heavy_cluster'].append(i)
        elif 0.3 < v['R'] < 0.8:
            clusters_1['semi_cluster'].append(i)
        elif 0.8 < v['R'] < 1.2:
            clusters_1['random'].append(i)
        elif 1.2 < v['R'] < 1.4:
            clusters_1['semi_regular'].append(i)
        elif 1.4 < v['R']:
            clusters_1['regular'].append(i)

    # Method 2: Generating clusters via KMeans using the regularity index R, the quartile coefficient of dispersion and the first eigenvalue
    df = pd.DataFrame.from_dict(instances, orient='index', columns=[
                                'optimal', 'mean', 'median', 'coeff_var', 'qdc', 'R', 'eigen1'])
    df_cluster = df[['R', 'qdc', 'eigen1']]

    std_scaler = StandardScaler()
    cluster = std_scaler.fit_transform(df_cluster.to_numpy())

    kmeans = KMeans(n_clusters=n_clusters_kmeans).fit(cluster)
    df_cluster['cluster'] = kmeans.labels_.astype(str)

    clusters_2 = [[] for k in range(n_clusters_kmeans)]
    for k, v in enumerate(kmeans.labels_):
        clusters_2[v].append(df.iloc[[k]].index[0])

    with open(output_filepath+'clusters.json', "w") as outfile:
        out = {'clusters_range': clusters_1, 'clusters_kmeans': clusters_2}
        json.dump(out, outfile, indent=4, default=str)
        f.close()

    if plot_cluster:
        fig = px.scatter_3d(df_cluster, x='R', y='qdc', z='eigen1',
                            color=df_cluster['cluster'], hover_name=df_cluster.index)
        fig.update_traces(marker_size=10)
        fig.show()
        fig.write_image(output_filepath+"clusters_kmeans.png",
                        scale=3, width=850, height=800)
