from scipy.optimize import OptimizeResult
import scipy.stats as ss
import scikit_posthocs as sp
import itertools as it
from statsmodels.sandbox.stats.multicomp import multipletests
from typing import Union
from problem import Problem
from tsp import TSP
from config import output_folder_prefix, params
import json
from pathlib import Path
import dill as pickle
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skopt.space import Space, Integer, Real
from skopt.plots import plot_convergence, plot_objective
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None


class Analyzer:
    # specify mode constants
    MODE_RUN = "run"
    MODE_EXPERIMENT = "exp"
    MODE_OPTIMIZE = "opt"

    def __init__(self, mode=MODE_RUN, obj_algorithm='hsppbo', results_path="output/", output_path=None) -> None:
        self.mode = mode
        self.path_prefix = output_folder_prefix[mode]
        self.results_path = results_path
        self.obj_algorithm = obj_algorithm
        self.output_path = output_path

    def create_results(self) -> None:
        results = {}
        if self.mode == self.MODE_RUN:
            pass
        elif self.mode == self.MODE_OPTIMIZE:
            pass
        elif self.mode == self.MODE_EXPERIMENT:
            pass
        self.create_result_log(results)

    def create_run_results(self, optimal_solution):
        io_file = self.create_file_wrapper("results.csv")
        # rpd_list = [((x[4] - optimal_solution) / optimal_solution) for x in self.log_list]
        # time_list =  [x[1] for x in self.log_list]

    def create_result_log(self, results: dict):
        """
        Create folder info about the relation between subdirectories and their respective parameters
        """
        io_file = self.create_file_wrapper("results.json")
        io_file.write(json.dumps(results, indent=4, default=str))
        io_file.close()

    def create_run_plot(self, result_nums: list[int], cmp='dynamic', aggr='problem', iteration_range=[1950, 2300], paths_dict=None):
        """
        Create a line plot over the quality of the current best solution over custom iteration range.
        Also, the mode of comparison and aggregation can be changed, so different optimizers, problems or dynamics can be compared and viewed in subplots.

        Args:
            result_nums (list[int]): List of the folder suffixes to be processed, e.g. [5,6] for exp_5 and exp_6.
            cmp (str, optional): Comparison identifier to compare the results against, has to be ["optimizer","dynamic","problem"] or None. Defaults to 'dynamic'.
            aggr (str, optional): Aggregation identifier to make subplots the results for each group, has to be ["optimizer","dynamic","problem"] or None. Defaults to 'problem'.
            paths_dict (dict, optional): Dict to provide the path of the needed data directly, instead of iterating over results. Used for recursion only. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        results = {}

        for n in result_nums:

            if paths_dict:
                folder = paths_dict
            else:
                # load folder dict and the relevant contents
                folder = self.load_result_folder(n)

            if 'sub' in folder:
                for sub_paths in folder.keys():
                    if type(folder[sub_paths]) is dict:
                        key, res = self.create_run_plot(
                            [n], paths_dict=folder[sub_paths])
                        results.setdefault(key, []).append(res)

            if self.mode == self.MODE_RUN:
                res = load_run(folder['run.csv'])
            elif self.mode == self.MODE_EXPERIMENT:
                res = load_avg_run(folder['avg_run.pkl'])
                res['best_solution'] = res['best_solution'].apply(np.mean)
            else:
                raise NotImplementedError(
                    "The mode %s currently does not support a run plot." % self.mode)

            info = get_info(folder['info.json'])
            res['cmp'] = get_key_parameter(cmp, info)
            opt_solution = get_optimal_solution(folder['info.json'])
            if opt_solution:
                res['best_solution'] = res['best_solution'].subtract(
                    opt_solution).divide(opt_solution)
            # append or create list for current aggregation key

            if paths_dict:
                return get_key_parameter(aggr, info), res
            else:
                results.setdefault(get_key_parameter(
                    aggr, info), []).append(res)

        fig = make_subplots(shared_xaxes=True, vertical_spacing=0.05,
                            rows=len(results), cols=1,
                            subplot_titles=(list(results.keys())))
        count = 1
        for res in results.values():
            res = pd.concat([r for r in res], ignore_index=True, sort=False)
            res = res.loc[(res['iteration'] < iteration_range[1]) &
                          (res['iteration'] >= iteration_range[0])]
            fe = px.line(res, x="iteration", y="best_solution", color="cmp" if cmp else None,
                         category_orders={"cmp": sorted(res['cmp'].unique())})
            for f in fe['data']:
                fig.add_trace(go.Scatter(f, showlegend=True if count == 1 else False,
                              legendgroup='group'), row=count, col=1)
            count += 1

        fig.update_annotations(font_size=14)
        fig.update_layout(legend_title_text=cmp, font_size=11, boxmode='group')
        fig.update_layout(
            title={
                'text': 'Quality of current best solution over iterations, shown for different %ss<br>(%s comparison over runs {%s})' %
                (aggr, cmp, ",".join(str(x) for x in result_nums)),
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        # results = pd.concat([r for r in results], ignore_index=True, sort=False)
        # results = results.loc[(results['iteration'] < 2300) & (results['iteration'] >= 1950)]
        # print(results)
        # fig = px.line(results, x="iteration", y="best_solution", color='cmp')

        fig.show()

    def create_param_boxplot(self, result_nums: list[int], cmp='optimizer', paths_dict=None):
        """
        Create a boxplot over the best parameter sets found during during the optimization process (here alpha, beta, the three weight and the detection threshold).
        Also, the mode of comparison can be changed, so different optimizers, problems or dynamics can be compared.

        Args:
            result_nums (list[int]): List of the folder suffixes to be processed, e.g. [5,6] for opt_5 and opt_6.
            cmp (str, optional): Comparison identifier to compare the results against, has to be ["optimizer","dynamic","problem"] or None. Defaults to 'optimizer'.
            paths_dict (dict, optional): Dict to provide the path of the needed data directly, instead of iterating over results. Used for recursion only. Defaults to None.
        """
        results = pd.DataFrame()

        if paths_dict:
            folder = paths_dict
            res = load_opt_best_params(folder['opt_best_params.csv'])
            info = get_info(folder['info.json'])
            key = get_key_parameter(cmp, info)
            res['cmp'] = key
            return res

        for n in result_nums:
            # load folder dict and the relevant contents
            folder = self.load_result_folder(n)

            if 'sub' in folder:
                for sub_paths in folder.keys():
                    if type(folder[sub_paths]) is dict:
                        res = self.create_param_boxplot(
                            [n], cmp=cmp, paths_dict=folder[sub_paths])
                        results = pd.concat(
                            [results, res], ignore_index=True, sort=False)
            else:
                res = load_opt_best_params(folder['opt_best_params.csv'])
                info = get_info(folder['info.json'])
                key = get_key_parameter(cmp, info)
                res['cmp'] = key
                results = pd.concat(
                    [results, res], ignore_index=True, sort=False)

        print(results)

        y1 = ["alpha", "beta"]
        y2 = ['w_pers_best', 'w_pers_prev', 'w_parent_best']
        if 'detection_threshold' in results.columns:
            y2.append('detection_threshold')

        fig = make_subplots(shared_yaxes=True,
                            rows=2, cols=1, vertical_spacing=0.1)
        fig1 = px.box(results, y=y1,
                      color="cmp" if cmp else None, points="all", category_orders={"cmp": sorted(res['cmp'].unique())})
        fig2 = px.box(
            results, y=y2, color="cmp" if cmp else None, points="all", category_orders={"cmp": sorted(res['cmp'].unique())})

        for f in fig1['data']:
            fig.add_trace(go.Box(f, showlegend=False,
                          legendgroup='group'), row=1, col=1)
        for f in fig2['data']:
            fig.add_trace(go.Box(f, legendgroup='group'), row=2, col=1)

        fig.update_layout(legend_title_text=cmp, boxmode='group')
        fig.update_layout(
            title={
                'text': 'Boxplot of best parameter sets (%s comparison over runs {%s})' %
                (cmp, ",".join(str(x) for x in result_nums)),
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        fig.show()

    def create_param_scatter_matrix(self, result_nums: list[int], cmp='optimizer', paths_dict=None):
        """
        Create a scatter matrix over the best parameter sets found during the optimization process (here alpha, beta, the three weight and the detection threshold).
        Also, the mode of comparison can be changed, so different optimizers, problems or dynamics can be compared.

        Args:
            result_nums (list[int]): List of the folder suffixes to be processed, e.g. [5,6] for opt_5 and opt_6.
            cmp (str, optional): Comparison identifier to compare the results against, has to be ["optimizer","dynamic","problem"] or None. Defaults to 'optimizer'.
            paths_dict (dict, optional): Dict to provide the path of the needed data directly, instead of iterating over results. Used for recursion only. Defaults to None.
        """
        results = pd.DataFrame()

        if paths_dict:
            folder = paths_dict
            res = load_opt_best_params(folder['opt_best_params.csv'])
            info = get_info(folder['info.json'])
            key = get_key_parameter(cmp, info)
            res['cmp'] = key
            return res

        for n in result_nums:
            # load folder dict and the relevant contents
            folder = self.load_result_folder(n)

            if 'sub' in folder:
                for sub_paths in folder.keys():
                    if type(folder[sub_paths]) is dict:
                        res = self.create_param_boxplot(
                            [n], cmp=cmp, paths_dict=folder[sub_paths])
                        results = pd.concat(
                            [results, res], ignore_index=True, sort=False)
            else:
                res = load_opt_best_params(folder['opt_best_params.csv'])
                info = get_info(folder['info.json'])
                key = get_key_parameter(cmp, info)
                res['cmp'] = key
                results = pd.concat(
                    [results, res], ignore_index=True, sort=False)

        y = ['alpha', 'beta', 'w_pers_best', 'w_pers_prev', 'w_parent_best']
        if 'detection_threshold' in results.columns:
            y.append('detection_threshold')

        fig = px.scatter_matrix(results,
                                dimensions=y,
                                color="cmp" if cmp else None)
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        fig.update_layout(legend_title_text=cmp)
        if cmp:
            txt = ('Scatter matrix of best parameter sets (%s comparison over runs {%s})' %
                   (cmp, ",".join(str(x) for x in result_nums)))
        else:
            txt = ('Parameter Scatter Matrix (over runs {%s})' %
                   (",".join(str(x) for x in result_nums)))
        fig.update_layout(
            title={
                'text': txt,
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )
        fig.show()

    def create_convergence_plot(self, result_nums: list[int], paths_dict=None):
        """
        Plot one or several convergence traces for each optimization method used, aggregating over all results provided.

        Args:
            result_nums (list[int]): List of the result folder suffixes to be processed, e.g. [5,6] for opt_5 and opt_6.
            paths_dict (dict, optional): Dict to provide the path of the needed data directly, instead of iterating over results. Used for recursion only. Defaults to None.
        """

        results = []
        problems = set()
        opt_solution = None

        if paths_dict:
            folder = paths_dict
            opt_solution = get_optimal_solution(folder['info.json'])
            info = get_info(folder['info.json'])
            opt = info['optimizer']
            problems.add(info['problem']['name'])
            return (opt, load_opt_result(folder, pickled=False))

        for n in result_nums:

            # load IO wrapper for accesing files from run
            folder = self.load_result_folder(n)

            if 'sub' in folder:
                for sub_paths in folder.keys():
                    if type(folder[sub_paths]) is dict:
                        results.append(self.create_convergence_plot(
                            [n], paths_dict=folder[sub_paths]))
            else:
                opt_solution = get_optimal_solution(folder['info.json'])
                info = get_info(folder['info.json'])
                opt = info['optimizer']
                problems.add(info['problem']['name'])
                results.append(
                    (opt, load_opt_result(folder, pickled=False)))

        if len(problems) == 1:
            problem = problems.pop()
        else:
            problem = None
            new_r = {}
            for r in results:
                new_r.setdefault(r[0], []).extend(r[1])
            results = []
            for k, v in new_r.items():
                results.append((k, v))

        if results:
            plot = plot_convergence(*results)
            if opt_solution:
                plot.set_ylim([0.00, 0.25])
                plot.set_xlim([8, 31])
                plot.set_ylabel(
                    r"relative difference to optimal solution quality after $n$ calls")
            if problem:
                plot.set_title(f'Convergence plot ({problem})')
            plot.legend(loc="upper right", prop={'size': 8}, numpoints=1)

            if self.output_path and problem:
                plt.savefig(
                    "/".join([self.output_path, f'convergence_{problem}.png']), dpi=300)
            elif self.output_path:
                plt.savefig(
                    "/".join([self.output_path, 'convergence_all.png']), dpi=300)
            else:
                plt.show()

    def create_partial_dep_plot(self, result_num: int, paths_dict=None, n_points=40):
        """
        Plot a 2-d matrix with so-called Partial Dependence plots of the objective function. 
        This shows the influence of each search-space dimension on the objective function.

        The diagonal shows the effect of a single dimension on the objective function,
        while the plots below the diagonal show the effect on the objective function when varying two dimensions.

        The Partial Dependence is calculated by averaging the objective value for a number of random samples in the search-space,
        while keeping one or two dimensions fixed at regular intervals.
        This averages out the effect of varying the other dimensions and shows the influence of one or two dimensions on the objective function.

        Note:
        The Partial Dependence plot is only an estimation of the surrogate model which in turn is only an estimation of the true objective function that has been optimized

        Args:
            result_num (int, optional): Result folder suffix to process.
            paths_dict (_type_, optional): Dict to provide the path of the needed data directly, instead of iterating over results. Used for recursion only. Defaults to None.
            n_points (int, optional): Number of points at which to evaluate the partial dependence along each dimension. Defaults to 40.
        """
        results = []

        if paths_dict:
            folder = paths_dict
        else:
            # load IO wrapper for accessing files from run
            folder = self.load_result_folder(result_num)

        if 'sub' in folder:
            for sub_paths in folder.keys():
                if type(folder[sub_paths]) is dict:
                    self.create_partial_dep_plot(
                        result_num, folder[sub_paths], n_points=n_points)
            return

        info = get_info(folder['info.json'])
        problem = info['problem']['name']
        dynamic_intensity = info['problem']['dynamic_props']['dynamic_intensity']
        opt = get_optimizer_type(folder['info.json'])

        results = load_opt_result(folder)

        dimensions = [d[0] for d in params[self.mode][self.obj_algorithm]]
        for i, res in enumerate(results):
            plot_objective(res, dimensions=dimensions, n_points=n_points)
            plt.suptitle('Partial Dependence (opt_%d, %s, %s [C=%d], run %d of %d)' %
                         (result_num, opt, problem, dynamic_intensity, i+1, len(results)))

        plt.show()

    def get_feature_importance(self, result_num: int, paths_dict=None) -> list[dict]:
        """
        Get the feature importance of a model built during the optimization process.
        Currently only forest and gradient decent optimizers are supported.

        Args:
            result_num (int, optional): Result folder suffix to process.
            paths_dict (_type_, optional): Dict to provide the path of the needed data directly, instead of iterating over results. Used for recursion only. Defaults to None.

        Returns:
            list[dict]: A list entry for each run within the result, containing a dict of all the parameters and their feature importance
        """
        param_importance = []

        if paths_dict:
            folder = paths_dict
        else:
            folder = self.load_result_folder(result_num)

        if 'sub' in folder:
            for sub_paths in folder.keys():
                if type(folder[sub_paths]) is dict:
                    self.get_feature_importance(
                        result_num, folder[sub_paths])
            return

        info = get_info(folder['info.json'])
        res = load_opt_result(folder, True)

        dim = [d[0] for d in params[self.mode][self.obj_algorithm]]

        for ir, r in enumerate(res):
            # load last model from opt run
            param_importance.append({})
            model = r.models[-1]
            if info['optimizer'] == 'gradient':
                reg = model.regressors_
                for id, d in enumerate(dim):
                    param_importance[ir][d] = np.mean(
                        [reg[x].feature_importances_[id]
                            for x in range(len(reg))]
                    )
            elif info['optimizer'] == 'forest':
                fi = model.feature_importances_
                for id, d in enumerate(dim):
                    param_importance[ir][d] = fi[id]
            else:
                raise NotImplementedError(
                    'The optimizer method %s does not support this function' % (info['optimizer']))

        return param_importance

    def get_convergence_stats(self, result_nums: list[int], start_iteration=11, avg_res=False) -> dict:
        """
        Calculating multiple statistic relevant to the convergence behavior of an optimization algorithm.
        Statistics are grouped by optimization algorithm and contain:
            - min: the absolute minimal solution quality found during optimization
            - auc: the area under the curve of a results convergence graph, averaged over all runs 
            - kwh + conover: Kruskal-Wallis test + Post-Hoc Conover-Iman test
            - mannwhitneyu: (only for h0-rejects of Conover's test) the bonferroni corrected p-value of the one-sided (lesser) Mann–Whitney U test

        Note: the often as similar to the Kruskal-Wallis test regarded Friedman test is not used, since assumes the values to be paired or dependent,
        which this data is not, since each measurement is taken from a differently initialized algorithm. 

        Args:
            result_nums (list[int]): Result folder suffix to process.
            start_iteration (int, optional): If the iterations are needed for calculations, this is the value from which it starts.
                                             Defaults to 11, because the first ten iterations are randomly sampled and do not depend on algorithm performance.
            avg_res (bool, optional): If all statistics shall be averaged for respective algorithm, instead of being lists. Defaults to False.

        Returns:
            dict: Dict containing statistics as mentioned above, grouped by optimization algo.
        """
        stats = {}
        tests = {}
        problems = set()

        for n in result_nums:
            # load IO wrapper for accesing files from run
            folder = self.load_result_folder(n)
            info = get_info(folder['info.json'])
            opt = info['optimizer']
            problems.add(info['problem']['name'])

            if opt not in stats:
                stats[opt] = {'auc': [], 'min': [],
                              'mean_mins': []}
            opt_solution = get_optimal_solution(folder['info.json'])
            res = load_opt_result(folder, pickled=False)
            n_calls = len(res[0].x_iters)
            iterations = range(start_iteration, n_calls + 1)
            mins = [[np.min(r.func_vals[:i]) for i in iterations]
                    for r in res]
            mean_x = np.mean(mins, axis=0)
            abs_min = np.min([(r.fun-opt_solution)/opt_solution for r in res])
            stats[opt]['auc'].append(metrics.auc(iterations, mean_x))
            stats[opt]['min'].append(abs_min)
            stats[opt]['mean_mins'].append(mean_x)

        mean_min_key_list = [x for x in stats.keys()]
        mean_min_val_list = [np.mean(x['mean_mins'], axis=0)
                             for x in stats.values()]
        kwh = ss.kruskal(*mean_min_val_list)
        tests['kwh'] = {'statistic': kwh[0], 'pvalue': kwh[1]}

        if kwh[1] <= 0.05:  # h0 reject of Kruskal-Wallis test is post-hoc checked via Conover-Iman test
            sp.posthoc_conover = posthoc_conover
            conover_t, conover_p = sp.posthoc_conover(mean_min_val_list, p_adjust='bonf')
            conover_p.columns, conover_p.index = mean_min_key_list, mean_min_key_list
            conover_t.columns, conover_t.index = mean_min_key_list, mean_min_key_list

            con_h0 = (conover_p < 0.05)
            # number of previous pairwise tests (conover) plus the to be executed mwu tests
            bonf_cor = len(mean_min_val_list)**2 + con_h0.to_numpy().sum()
            tests['conover_t'] = conover_t.to_dict()
            tests['conover_p'] = conover_p.to_dict()

            tests['mwu'] = {}
            for k, v in con_h0.to_dict().items():
                tests['mwu'][k] = {}
                for k2, v2 in v.items():
                    if k is not k2 and v2:
                        x = np.mean(stats[k]['mean_mins'], axis=0)
                        y = np.mean(stats[k2]['mean_mins'], axis=0)
                        mwu = ss.mannwhitneyu(
                            x[start_iteration:], y[start_iteration:], alternative='less')
                        tests['mwu'][k][k2] = {
                            'statistic': mwu[0], 'pvalue': mwu[1]*bonf_cor if mwu[1] < (0.05/bonf_cor) else mwu[1]}

        for k, v in stats.items():
            # v.pop('mean_mins')
            if avg_res or len(problems) == 1:
                v['auc'] = np.mean(v['auc'])
                v['min'] = np.mean(v['min'])
        if len(problems) == 1:
            stats['problem'] = problems.pop()

        return stats, tests

    def create_func_opt_boxplot(self, result_nums: list[int], start_iteration=11) -> None:
        """
        Creating a boxplot of the relative difference to optimal solution quality of all minimal func values and the AUC of the solution quality convergence graph.
        Values are being gathered across all provided results and x axes is split by optimization algorithm.

        Args:
            result_nums (list[int]): Result folder suffix to process.
            start_iteration (int, optional): If the iterations are needed for calculations, this is the value from which it starts.
                                             Defaults to 11, because the first ten iterations are randomly sampled and do not depend on algorithm performance.
        """
        func_results = {}
        auc_results = {}
        for n in result_nums:
            # load IO wrapper for accesing files from run
            folder = self.load_result_folder(n)
            opt = get_optimizer_type(folder['info.json'])
            opt_solution = get_optimal_solution(folder['info.json'])
            res = load_opt_result(folder, pickled=False)

            n_calls = len(res[0].x_iters)
            iterations = range(start_iteration, n_calls + 1)
            mins = [[np.min(r.func_vals[:i]) for i in iterations]
                    for r in res]
            auc_results.setdefault(opt, []).extend(
                [metrics.auc(iterations, vals) for vals in mins])
            func_results.setdefault(opt, []).extend(
                [(r.fun-opt_solution)/opt_solution for r in res])

        fig = make_subplots(shared_xaxes=True,
                            rows=2, cols=1, vertical_spacing=0.05,
                            subplot_titles=("Relative difference to optimal solution quality<br>of all minimal function values ", "AUC of solution quality convergence graph"))
        fig1 = px.box(pd.DataFrame(func_results))
        fig2 = px.box(pd.DataFrame(auc_results))
        fig.add_trace(go.Box(fig1['data'][0]), row=1, col=1)
        fig.add_trace(go.Box(fig2['data'][0]), row=2, col=1)
        fig.update_layout(margin={'l': 10, 'r': 10, 't': 45, 'b': 10})
        fig.update_annotations(font_size=14)
        if self.output_path:
            fig.write_image("/".join([self.output_path, 'convergence_stats_boxplot.png']),
                            format="png", width=400, height=750, scale=2)
        else:
            fig.show()

    def load_result_folder(self, result_num: int) -> Path:
        """
        Load the content of an output folder

        Args:
            result_num (int): Folder suffix to load, e.g. 5 for opt_5

        Returns:
            Generator[Path, None, None]: Generator provided by Pathlib to access its contents 
        """
        path = "".join(
            [self.results_path, self.path_prefix, str(result_num), '/'])
        path_obj = Path(path)
        if path_obj.exists():
            files = {}
            for p in path_obj.iterdir():
                if p.name == 'folder_info.json':
                    files['sub'] = True
                if p.is_dir():
                    files[p.name] = {x.name: x.as_posix()
                                     for x in Path(p.as_posix()).iterdir()}
                elif p.is_file():
                    files[p.name] = p.as_posix()

            return files
        return None

    def create_file_wrapper(self, result_num: int, filename: str, mode='a') -> None:
        """
        Initializes a file wrapper for logging

        Args:
            filename (str): Name of the file that should be created

        Returns:
            TextIOWrapper: Wrapper for the opened logging file
        """
        return open("".join(self.results_path, self.path_prefix, str(result_num), '/', filename), mode)


def load_opt_best_params(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=';')
        df = df.loc[:, ~df.columns.isin(['run', 'func_val'])]
    except:
        raise FileExistsError(
            'File "opt_best_params.csv" not found in current folder')

    return df


def load_avg_run(path: str) -> pd.DataFrame:
    try:
        return pd.read_pickle(path)
    except:
        raise FileExistsError(
            'File from provided path could not be loaded: %s' % path)


def load_run(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=";")
    except:
        raise FileExistsError(
            'File from provided path could not be loaded: %s' % path)


def load_opt_result(folder: dict, pickled=True, normalize=True) -> list[OptimizeResult]:
    """
    Load the results of the optimizing runs, generally being of the OptimizeResult class

    Args:
        folder (dict):              Dict of the filenames and their locations
        pickled (bool, optional):   If the results are provided in a pickled or unpickled form.
                                    Pickled preferred, json eventually deprecated. Defaults to True.
        normalize (bool, optional): Set if the function results should be normalized by the optimum of the problem instance

    Returns:
        list[OptimizeResult]: List of OptimizeResult classes, being results of optimization runs
    """

    res = []

    opt_solution = get_optimal_solution(folder['info.json'])

    for k, v in folder.items():
        if 'opt_log' in k and 'pkl' in k and pickled:
            with open(v, 'rb') as f:
                log = pickle.load(f)
                f.close()
            res.append(log)
        if 'opt_log' in k and 'json' in k and not pickled:
            with open(v, 'r') as f:
                log = json.load(
                    f, object_hook=lambda d: OptimizeResult(**d))
                f.close()
                if opt_solution and normalize:
                    log.func_vals = np.array(
                        [(x-opt_solution)/opt_solution for x in log.func_vals])
                else:
                    log.func_vals = np.array(log.func_vals)
            res.append(log)

    return res


def get_key_parameter(cmp: str, info: dict) -> str:
    """
    Returns the corresponding info for the provided comparison identifier to add to the data.

    Args:
        cmp (str): Comparison identifier to compare the results against
        info (dict): The meta info of the run currently processed

    Raises:
        ValueError: Raised if value for cmp is not "optimizer","dynamic","problem" or None

    Returns:
        str: Comparable info about the current run, e.g. the optimizer type, the dynamic intensity used or the problem.
    """
    if cmp is None:
        key = ''
    elif cmp == 'optimizer':
        key = info['optimizer']
    elif cmp == 'dynamic':
        key = info['problem']['dynamic_props']['dynamic_intensity']
    elif cmp == 'problem':
        key = info['problem']['name']
    else:
        raise ValueError(
            'Invalid value for parameter cmp. Chose from ["optimizer","dynamic","problem" or None]')
    return key


def get_optimizer_type(path: str) -> str:
    with open(path, 'r') as f:
        info = json.load(f)
        f.close()
    return info['optimizer']


def get_info(path: str) -> str:
    with open(path, 'r') as f:
        info = json.load(f)
        f.close()
    return info


def get_optimal_solution(path: str) -> int:
    problem = get_info(path)['problem']['name']
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

def posthoc_conover(
        a: Union[list, np.ndarray, pd.DataFrame],
        val_col: str = None,
        group_col: str = None,
        p_adjust: str = None,
        sort: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:

    def compare_conover(i, j):
        diff = np.abs(x_ranks_avg.loc[i] - x_ranks_avg.loc[j])
        B = (1. / x_lens.loc[i] + 1. / x_lens.loc[j])
        D = (n - 1. - h_cor) / (n - x_len)
        t_value = diff / np.sqrt(S2 * B * D)
        p_value = 2. * ss.t.sf(np.abs(t_value), df=n-x_len)
        return t_value, p_value

    x, _val_col, _group_col = sp.__convert_to_df(a, val_col, group_col)
    x = x.sort_values(by=[_group_col, _val_col], ascending=True) if sort else x

    n = len(x.index)
    x_groups_unique = x[_group_col].unique()
    x_len = x_groups_unique.size
    x_lens = x.groupby(_group_col)[_val_col].count()

    x['ranks'] = x[_val_col].rank()
    x_ranks_avg = x.groupby(_group_col)['ranks'].mean()
    x_ranks_sum = x.groupby(_group_col)['ranks'].sum()

    # ties
    vals = x.groupby('ranks').count()[_val_col].values
    tie_sum = np.sum(vals[vals != 1] ** 3 - vals[vals != 1])
    tie_sum = 0 if not tie_sum else tie_sum
    x_ties = np.min([1., 1. - tie_sum / (n ** 3. - n)])

    h = (12. / (n * (n + 1.))) * np.sum(x_ranks_sum**2 / x_lens) - 3. * (n + 1.)
    h_cor = h / x_ties

    if x_ties == 1:
        S2 = n * (n + 1.) / 12.
    else:
        S2 = (1. / (n - 1.)) * (np.sum(x['ranks'] ** 2.) - (n * (((n + 1.)**2.) / 4.)))

    vs = np.zeros((x_len, x_len))
    tvs = np.zeros((x_len, x_len))
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0
    tvs[:, :] = 0

    combs = it.combinations(range(x_len), 2)

    for i, j in combs:
        tvs[i, j], vs[i, j] = compare_conover(x_groups_unique[i], x_groups_unique[j])

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]
    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)

    return pd.DataFrame(tvs, index=x_groups_unique, columns=x_groups_unique), pd.DataFrame(vs, index=x_groups_unique, columns=x_groups_unique)
