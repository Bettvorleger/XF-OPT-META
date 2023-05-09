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
import re
from math import floor
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

        if self.output_path:
            Path(self.output_path).mkdir(parents=True, exist_ok=True)

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

    def create_run_plot(self, result_nums: list[int], cmp='dynamic', aggr='problem', y_value='best_solution', y_value_range=None, iteration_range=[1990, 2399], alt_results_path=None, custom_keyorder=None, paths_dict=None):
        """
        Create a line plot over the quality of the current best solution over custom iteration range.
        Also, the mode of comparison and aggregation can be changed, so different optimizers, problems or dynamics can be compared and viewed in subplots.

        Args:
            result_nums (list[int]): List of the folder suffixes to be processed, e.g. [5,6] for exp_5 and exp_6.
            cmp (str, optional): Comparison identifier to compare the results against, has to be ["optimizer","dynamic","problem","folder"] or None. Defaults to 'dynamic'.
            aggr (str, optional): Aggregation identifier to make subplots the results for each group, has to be ["optimizer","dynamic","problem"] or None. Defaults to 'problem'.
            custom_keyorder (list[str], optional): List of order in which to display the comparison parameter's plots or legend items. Defaults to None.
            paths_dict (dict, optional): Dict to provide the path of the needed data directly, instead of iterating over results. Used for recursion only. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        results = {}
        results_info = {}

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
                res[y_value] = res[y_value].apply(np.mean)
            else:
                raise NotImplementedError(
                    "The mode %s currently does not support a run plot." % self.mode)

            info = get_info(folder['info.json'])
            opt_solution = get_optimal_solution(folder['info.json'])
            if opt_solution and y_value == 'best_solution':
                res['best_solution'] = res['best_solution'].subtract(
                    opt_solution).divide(opt_solution)
            # append or create list for current aggregation key

            if paths_dict:
                return get_key_parameter(aggr, info), res
            elif cmp == 'folder':
                key = re.findall(r"_[^_]+$", self.results_path)[0][1:-1]
                results.setdefault(get_key_parameter(
                    aggr, info), {}).setdefault(key, []).append(res)
                results_info.setdefault(get_key_parameter(
                    aggr, info), {}).setdefault(key, []).append(info)
            else:
                results.setdefault(get_key_parameter(
                    aggr, info), {}).setdefault(get_key_parameter(cmp, info), []).append(res)
                results_info.setdefault(get_key_parameter(
                    aggr, info), {}).setdefault(get_key_parameter(cmp, info), []).append(info)

        if cmp == 'folder':
            for n in result_nums:
                folder = self.load_result_folder(
                    n, results_path=alt_results_path)
                res = load_avg_run(folder['avg_run.pkl'])

                res[y_value] = res[y_value].apply(np.mean)
                info = get_info(folder['info.json'])

                opt_solution = get_optimal_solution(folder['info.json'])
                if opt_solution and y_value == 'best_solution':
                    res['best_solution'] = res['best_solution'].subtract(
                        opt_solution).divide(opt_solution)

                key = re.findall(r"_[^_]+$", alt_results_path)[0][1:-1]
                results.setdefault(get_key_parameter(
                    aggr, info), {}).setdefault(key, []).append(res)
                results_info.setdefault(get_key_parameter(
                    aggr, info), {}).setdefault(key, []).append(info)

        margin = dict(l=0, r=0, t=0, b=0)
        height = 350
        width = 700
        note = f'{result_nums[0]}_to_{result_nums[-1]}'

        max_y_value = 0
        min_y_value = float('inf')

        if aggr == 'problem' and custom_keyorder:
            results = {k: results[k] for k in custom_keyorder if k in results}

        if aggr and cmp:
            sub_prefix = 'C=' if aggr == 'dynamic' else ''
            fig = make_subplots(shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.03, horizontal_spacing=0.01,
                                rows=int(len(results)/2) if len(results) > 3 else 1, cols=2 if len(results) > 3 else 3,
                                subplot_titles=(list(sub_prefix + str(x) for x in results.keys())))

            margin = dict(l=0, r=0, t=25, b=0)
            height = floor(len(results)/2)*150+300 if len(results) > 3 else 250
            width = 600

            if len(results) > 1:
                fig.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="right",
                    x=1
                ))

            for ir, res in enumerate(results.items()):
                for i, item in enumerate(res[1].items()):

                    if len(item[1]) == 1:
                        r = item[1][0]
                    else:
                        r = pd.DataFrame()
                        r['iteration'] = item[1][0]['iteration']
                        for it, t in enumerate(item[1]):
                            r[it] = t[y_value]
                        r[y_value] = r.drop('iteration', axis=1).mean(axis=1)

                    r = r.loc[(r['iteration'] < iteration_range[1]) &
                              (r['iteration'] >= iteration_range[0])]
                    max_y_value = r[y_value].quantile(
                        0.95) if max_y_value < r[y_value].quantile(0.95) else max_y_value
                    min_y_value = r[y_value].min(
                    ) if min_y_value > r[y_value].min() else min_y_value

                    fig.add_trace(go.Scatter(x=r['iteration'], y=r[y_value], name=item[0], line=dict(color=px.colors.qualitative.Plotly[i]),
                                             showlegend=True if ir == 0 else False, legendgroup=item[0]), row=floor(ir/2)+1 if len(results) > 3 else 1, col=ir % 2+1 if len(results) > 3 else ir+1)

                    if len(results) <= 3:
                        xaxis_title = ''
                        if y_value == 'best_solution':
                            xaxis_title='RPD'
                        elif y_value == 'swaps':
                            xaxis_title='Swaps'
                        fig.update_layout(
                            yaxis_title=xaxis_title
                        )

                    if cmp == 'dynamic' and aggr == 'problem':
                        if y_value == 'swaps':
                            theta = results_info[res[0]][item[0]
                                                         ][0]['hsppbo']['detection_threshold']
                            sce_count = results_info[res[0]
                                                     ][item[0]][0]['tree']['num_sce_nodes']
                            key = re.findall(
                                r"_[^_]+$", self.results_path)[0][1:-1]
                            fig.add_shape(
                                type='line', line=dict(dash='dash', color=px.colors.qualitative.Plotly[i] if key == 'HPO' else 'black'),
                                x0=iteration_range[0], x1=iteration_range[1], y0=theta*sce_count, y1=theta*sce_count, row=floor(ir/2)+1 if len(results) > 3 else 1, col=ir % 2+1 if len(results) > 3 else ir+1
                            )
        elif cmp:

            fig = go.Figure()
            results = dict(*results.values())
            results_info = dict(*results_info.values())
            width = 450

            for k, v in results.items():
                df = pd.DataFrame()
                df['iteration'] = v[0]['iteration']
                for i, r in enumerate(v):
                    df[i] = r[y_value]
                df[y_value] = df.drop('iteration', axis=1).mean(axis=1)

                df = df.loc[(df['iteration'] < iteration_range[1])
                            & (df['iteration'] >= iteration_range[0])]
                max_y_value = df[y_value].quantile(
                    0.95) if max_y_value < df[y_value].quantile(0.95) else max_y_value
                min_y_value = df[y_value].min(
                ) if min_y_value > df[y_value].min() else min_y_value

                fig.add_trace(go.Scatter(x=df['iteration'], y=df[y_value], name=k,
                                         showlegend=True))
                
                yaxis_title = ''
                if y_value == 'best_solution':
                    yaxis_title='RPD'
                elif y_value == 'swaps':
                    yaxis_title='Swaps'
                fig.update_layout(
                    xaxis_title='Iteration',
                    yaxis_title=yaxis_title
                )

                key = re.findall(r"_[^_]+$", self.results_path)[0][1:-1]
                if y_value == 'swaps':
                    width = 400
                    if key != 'HPO':
                        theta = results_info[k][0]['hsppbo']['detection_threshold']
                        sce_count = results_info[k][0]['tree']['num_sce_nodes']
                        fig.add_shape(type='line', line=dict(
                            dash='dash', color='black'), x0=iteration_range[0], x1=iteration_range[1], y0=theta*sce_count, y1=theta*sce_count)

            fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="right",
                x=1
            ))
            
        elif aggr:
            results = {k: list(*v.values()) for k, v in results.items()}

            fig = make_subplots(shared_xaxes=True, vertical_spacing=0.05,
                                rows=len(results), cols=1,
                                subplot_titles=(list(results.keys())))
            for ir, res in enumerate(results.values()):
                df = pd.DataFrame()
                df['iteration'] = res[0]['iteration']
                for i, r in enumerate(res):
                    df[i] = r[y_value]
                df[y_value] = df.drop('iteration', axis=1).mean(axis=1)
                df = df.loc[(df['iteration'] < iteration_range[1]) &
                            (df['iteration'] >= iteration_range[0])]
                max_y_value = df[y_value].quantile(
                    0.95) if max_y_value < df[y_value].quantile(0.95) else max_y_value
                min_y_value = df[y_value].min(
                ) if min_y_value > df[y_value].min() else min_y_value

                fig.add_trace(go.Scatter(
                    x=df['iteration'], y=df[y_value], showlegend=False, legendgroup=ir, line=dict(color=px.colors.qualitative.Plotly[0])), row=ir+1, col=1)
        else:
            fig = go.Figure()

            results = list(*dict(*results.values()).values())
            df = pd.DataFrame()
            df['iteration'] = results[0]['iteration']
            for i, r in enumerate(results):
                df[i] = r[y_value]
            df[y_value] = df.drop('iteration', axis=1).mean(axis=1)
            df = r.loc[(df['iteration'] < iteration_range[1]) &
                       (df['iteration'] >= iteration_range[0])]
            max_y_value = df[y_value].quantile(
                0.95) if max_y_value < df[y_value].quantile(0.95) else max_y_value
            min_y_value = df[y_value].min(
            ) if min_y_value > df[y_value].min() else min_y_value

            fig.add_trace(go.Scatter(x=df['iteration'], y=df[y_value]))
            
            yaxis_title = ''
            if y_value == 'best_solution':
                yaxis_title='RPD'
            elif y_value == 'swaps':
                yaxis_title='Swaps'
            fig.update_layout(
                xaxis_title='Iteration',
                yaxis_title=xaxis_title
            )

        if cmp:
            if cmp == 'dynamic':
                fig.update_layout(legend_title_text='Dynamic intensity C')
            elif cmp == 'problem':
                fig.update_layout(legend_title_text='Problem instance')
            elif cmp == 'folder':
                fig.update_layout(legend_title_text='Parameter Set Group')

        if 'xaxis2' in fig.layout:
            if aggr and len(results) > 3:
                for i in range(len(results)):
                    fig['layout'][f'xaxis{i+1}'].update(nticks=10)
        else:
            fig['layout']['xaxis'].update(nticks=10)

        if 'yaxis2' in fig.layout:
            if aggr and len(results) > 3:
                for i in range(len(results)):
                    fig['layout'][f'yaxis{i+1}'].update(nticks=10)
            if y_value_range:
                for i in range(len(results)):
                    fig['layout'][f'yaxis{i+1}'].update(
                        range=[y_value_range[0], y_value_range[1]])
            else:
                for i in range(len(results)):
                    fig['layout'][f'yaxis{i+1}'].update(
                        range=[min_y_value*0.9, max_y_value])
        else:
            if aggr and len(results) > 3:
                fig['layout']['yaxis'].update(nticks=10)
            if y_value_range:
                fig.update_layout(
                    yaxis_range=[y_value_range[0], y_value_range[1]])
            else:
                fig.update_layout(
                    yaxis_range=[min_y_value*0.8, max_y_value*1.1])

        if aggr and len(results) > 3 and 'yaxis2' in fig.layout:
            for i in range(len(results)):
                fig['layout'][f'yaxis{i+1}'].update(nticks=10)
        else:
            fig['layout']['yaxis'].update(nticks=10)

        key = re.findall(r"_[^_]+$", self.results_path)[0][1:-1]
        fig.update_annotations(font_size=14)
        fig.update_layout(font_size=11, boxmode='group')
        fig.update_layout(
            title={
                'text': f'Quality of current best solution over iterations, shown for different {aggr}s<br>({cmp} comparison from run {result_nums[0]} to {result_nums[-1]}, {key})',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        if self.output_path:
            fig.update_annotations(font_size=15)
            fig.update_layout(title=None, font_family="Helvetica",
                              font_size=15, margin=margin)
            fig.write_image("/".join([self.output_path, f'run_plot_cmp_{cmp}_aggr_{aggr}_y_{y_value}_{note}_{key}.svg']),
                            format="svg", width=width, height=height)
        else:
            fig.show()

    def create_pr_plot(self, result_nums: list[int], cmp='dynamic', grouped=True, custom_keyorder=None):
        """
        Generate a precision-recall scatter-plot from the average of all experimentation runs in an exp-folder.
        Comparable by custom parameters and either grouped in one plot or in multiple separate plots.

        Args:
            result_nums (list[int]): List of the folder suffixes to be processed, e.g. [5,6] for exp_5 and exp_6.
            cmp (str, optional): Comparison identifier to compare the results against, has to be ["optimizer","dynamic","problem"] or None. Defaults to 'dynamic'.
            grouped (bool, optional): Whether or not to plot the compared parameter in the same graphic. Defaults to True.
            custom_keyorder (list[str], optional): List of order in which to display the comparison parameter's plots or legend items. Defaults to None.

        Raises:
            NotImplementedError: Only experimentation mode supported.
        """

        results = {}
        results_info = {}
        for n in result_nums:
            folder = self.load_result_folder(n)

            if self.mode == self.MODE_EXPERIMENT:
                res = load_avg_run(folder['avg_run.pkl'])
            else:
                raise NotImplementedError(
                    "The mode %s currently does not support a run plot." % self.mode)

            info = get_info(folder['info.json'])
            results.setdefault(get_key_parameter(
                cmp, info), []).append(res)
            results_info.setdefault(get_key_parameter(
                cmp, info), []).append(info)

        stats = {}
        for k, v in results.items():
            stats[k] = []
            for ir, r in enumerate(v):
                r = r['reaction'].apply(pd.Series)
                for col in r:
                    positives, negatives, trigger_points = get_detection_info(
                        results_info[k][ir])
                    entry = calc_reset_accuracy(
                        r[col], trigger_points, positives, negatives)

                    entry['cmp'] = k
                    stats[k].append(entry)

        margin = dict(l=0, r=0, t=0, b=0)

        if cmp == 'detection_threshold':
            l = list()
            for v in stats.values():
                for el in v:
                    l.append(el)
            df = pd.DataFrame(l)

            fig = go.Figure()
            fig.add_trace(go.Histogram2d(
                x=df['recall'], y=df['precision'], z=df['cmp'], histfunc='avg', colorscale='Viridis', xbins=dict(start=-0.1, end=1.1, size=0.1), ybins=dict(start=-0.1, end=1.1, size=0.1)))
            fig.add_trace(go.Scatter(
                x=df['recall'], y=df['precision'], mode='markers', showlegend=False,
                marker=dict(
                    symbol='circle',
                    opacity=0.6,
                    color='white',
                    size=7,
                    line=dict(width=1),
                )
            ))
            fig.update_layout(
                xaxis_title='Recall',
                yaxis_title='Precision',
                width=450, height=400
            )
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=1-df['n'][0]/600, y1=1-df['n'][0]/600
            )
            fig['layout']['xaxis'].update(range=[-0.1, 1.1])
            fig['layout']['yaxis'].update(range=[-0.1, 1.1])

        else:
            if grouped:
                fig = go.Figure()
                for k, v in stats.items():
                    df = pd.DataFrame.from_dict(v)
                    df = df.sort_values(by=['recall'])
                    fig.add_trace(go.Scatter(
                        x=df['recall'], y=df['precision'], name=k, mode='markers'))

                fig.add_shape(
                    type='line', line=dict(dash='dash'),
                    x0=0, x1=1, y0=1-df['n'][0]/600, y1=1-df['n'][0]/600
                )
                fig.update_layout(
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    xaxis=dict(constrain='domain'),
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                    width=450, height=425
                )
                fig['layout']['xaxis'].update(range=[-0.05, 1.05])
                fig['layout']['yaxis'].update(range=[-0.05, 1.05])
            else:
                if cmp == 'problem' and custom_keyorder:
                    stats = {k: stats[k]
                             for k in custom_keyorder if k in stats}

                sub_prefix = 'C=' if cmp == 'dynamic' else ''
                fig = make_subplots(shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.03, horizontal_spacing=0.01,
                                    rows=int(len(stats)/2) if len(stats) > 3 else 1, cols=2 if len(stats) > 3 else 3,
                                    subplot_titles=(list(sub_prefix + str(x) for x in stats.keys())))
                for iv, v in enumerate(stats.values()):
                    df = pd.DataFrame.from_dict(v)
                    df = df.sort_values(by=['recall'])
                    fig.add_shape(
                        type='line', line=dict(dash='dash'),
                        x0=0, x1=1, y0=1-df['n'][0]/600, y1=1-df['n'][0]/600, row=floor(iv/2)+1 if len(stats) > 3 else 1, col=iv % 2+1 if len(stats) > 3 else iv+1
                    )
                    fig.add_trace(go.Scatter(
                        x=df['recall'], y=df['precision'], mode='markers', showlegend=False, legendgroup=iv, line=dict(color=px.colors.qualitative.Plotly[0])), row=floor(iv/2)+1 if len(stats) > 3 else 1, col=iv % 2+1 if len(stats) > 3 else iv+1)
                if cmp == 'dynamic':
                    fig.update_layout(
                        xaxis2_title='Recall',
                        yaxis_title='Precision',
                        width=800, height=300,
                    )
                    margin = dict(l=0, r=0, t=25, b=0)
                elif cmp == 'problem':
                    fig.update_layout(
                        width=400, height=1000,
                    )
                    margin = dict(l=0, r=0, t=25, b=0)
                else:
                    fig.update_layout(
                        width=600, height=600,
                        autosize=False,
                    )
                for i in range(len(stats)):
                    fig['layout'][f'xaxis{i+1}'].update(range=[-0.05, 1.05])
                    fig['layout'][f'yaxis{i+1}'].update(range=[-0.05, 1.05])

        key = re.findall(r"_[^_]+$", self.results_path)[0][1:-1]
        if self.output_path:
            fig.update_annotations(font_size=15)
            fig.update_layout(title=None, font_family="Helvetica",
                              font_size=15, margin=margin)
            fig.write_image("/".join([self.output_path, f'pr_curve_cmp_{cmp}_grouped_{grouped}_{key}.svg']),
                            format="svg")
        else:
            fig.update_layout(
                title={
                    'text': f'Precision-Recall-Plot ({key})',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'}
            )
            fig.show()

    def create_fp_theta_plot(self, result_nums: list[int]):
        """
        Generate a false-positives over theta (detection threshold) scatter-plot from the average of all experimentation runs in an exp-folder.
        Comparable by custom parameters and either grouped in one plot or in multiple separate plots.

        Args:
            result_nums (list[int]): List of the folder suffixes to be processed, e.g. [5,6] for exp_5 and exp_6.
        Raises:
            NotImplementedError: Only experimentation mode supported.
        """

        results = []
        results_info = []
        for n in result_nums:
            folder = self.load_result_folder(n)

            if self.mode == self.MODE_EXPERIMENT:
                res = load_avg_run(folder['avg_run.pkl'])
            else:
                raise NotImplementedError(
                    "The mode %s currently does not support a run plot." % self.mode)

            info = get_info(folder['info.json'])
            results.append(res)
            results_info.append(info)

        stats = []
        for ires, res in enumerate(results):
            r = res['reaction'].apply(pd.Series)
            for col in r:
                positives, negatives, trigger_points = get_detection_info(
                    results_info[ires])
                entry = calc_reset_accuracy(
                    r[col], trigger_points, positives, negatives)

                entry['theta'] = results_info[ires]['hsppbo']['detection_threshold']
                stats.append(entry)

        df = pd.DataFrame(stats)
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['theta'], y=df['fp'], showlegend=False, mode='markers'))
        fig.update_xaxes(
            nticks=10
        )
        fig.update_yaxes(type="log")
        fig.update_layout(
            yaxis_title='False Positives',
            xaxis_title='Detection Threshold',
            width=400, height=400
        )

        key = re.findall(r"_[^_]+$", self.results_path)[0][1:-1]
        if self.output_path:
            fig.update_annotations(font_size=15)
            fig.update_layout(title=None, font_family="Helvetica",
                              font_size=15, margin=dict(l=0, r=0, t=0, b=0))
            fig.write_image("/".join([self.output_path, f'fp_theta_plot_{key}.svg']),
                            format="svg")
        else:
            fig.update_layout(
                title={
                    'text': f'Precision-Recall-Plot ({key})',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'}
            )
            fig.show()

    def create_reset_accuracy_csv(self, result_nums: list[int], cmp='dynamic', aggr='problem', all_stats=False, custom_keyorder=None):
        """
        Export a csv of all statistics concerning the reset accuracy of the metaheuristic or only the F1-score, 
        averaged over all experimentation runs and optionally compared and/or aggregated by a custom parameter.

        Args:
            result_nums (list[int]): List of the folder suffixes to be processed, e.g. [5,6] for exp_5 and exp_6.
            cmp (str, optional): Comparison identifier to compare the results against, has to be ["optimizer","dynamic","problem"] or None. Defaults to 'dynamic'.
            aggr (str, optional): Aggregation identifier to make subplots the results for each group, has to be ["optimizer","dynamic","problem"] or None. Defaults to 'problem'.
            all_stats (bool, optional): Whether to export all metrics or only the F1-score (influences table structure). Defaults to False.
            custom_keyorder (list[str], optional): List of order in which to display the comparison parameter's plots or legend items. Defaults to None.

        Raises:
            NotImplementedError: Only experimentation mode supported.
        """

        results = {}
        results_info = {}
        for n in result_nums:
            folder = self.load_result_folder(n)

            if self.mode == self.MODE_EXPERIMENT:
                res = load_avg_run(folder['avg_run.pkl'])
            else:
                raise NotImplementedError(
                    "The mode %s currently does not support a run plot." % self.mode)

            info = get_info(folder['info.json'])
            results.setdefault(get_key_parameter(
                aggr, info), {}).setdefault(get_key_parameter(cmp, info), []).append(res)
            results_info.setdefault(get_key_parameter(
                aggr, info), {}).setdefault(get_key_parameter(cmp, info), []).append(info)

        stats = {}
        for ak, av in results.items():
            stats[ak] = {}
            for k, v in av.items():
                stats[ak][k] = {}
                avg_stats = []
                for ir, r in enumerate(v):
                    r = r['reaction'].apply(pd.Series)
                    for col in r:
                        positives, negatives, trigger_points = get_detection_info(
                            results_info[ak][k][ir])
                        entry = calc_reset_accuracy(
                            r[col], trigger_points, positives, negatives)

                        if all_stats:
                            avg_stats.append(entry)
                        else:
                            avg_stats.append(entry['f1'])
                if all_stats:
                    stats[ak][k] = pd.DataFrame(
                        avg_stats).mean(axis=0).to_dict()
                    avg = pd.DataFrame(avg_stats)['f1'].mean(axis=0)
                    error = pd.DataFrame(avg_stats)['f1'].sem(axis=0)
                    stats[ak][k]['f1'] = conv2siunitx(avg, error)
                else:
                    avg = pd.DataFrame(avg_stats).mean(axis=0).values[0]
                    error = pd.DataFrame(avg_stats).sem(axis=0).values[0]
                    stats[ak][k] = conv2siunitx(avg, error)

        if aggr == 'problem' and custom_keyorder:
            stats = {k: stats[k] for k in custom_keyorder if k in stats}

        key = re.findall(r"_[^_]+$", self.results_path)[0][1:-1]
        if all_stats:
            reform = {(outerKey, innerKey): values for outerKey, innerDict in stats.items()
                      for innerKey, values in innerDict.items()}
            # print(pd.DataFrame(reform).T.style.to_latex())
            with open("/".join([self.output_path, f'reset_accuracy_stats_{cmp}_{aggr}_{key}.csv']), 'w') as out:
                out.write(pd.DataFrame(reform).T.to_csv())
        else:
            with open("/".join([self.output_path, f'reset_accuracy_f1_{cmp}_{aggr}_{key}.csv']), 'w') as out:
                out.write(pd.DataFrame(stats).T.to_csv())
            # print(pd.DataFrame(stats).T.style.to_latex())

    def create_param_boxplot(self, result_nums: list[int], cmp='optimizer', paths_dict=None, export_values=False):
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

        y1 = ["alpha", "beta"]
        y2 = ['w_pers_best', 'w_pers_prev', 'w_parent_best']
        if 'detection_threshold' in results.columns:
            y2.append('detection_threshold')

        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.1)
        fig1 = px.box(results, y=y1,
                      color="cmp" if cmp else None)
        fig2 = px.box(
            results, y=y2, color="cmp" if cmp else None)

        for f in fig1['data']:
            fig.add_trace(go.Box(f, showlegend=False,
                          legendgroup='group', boxpoints=None if cmp else 'all'), row=1, col=1)
        for f in fig2['data']:
            fig.add_trace(go.Box(f, showlegend=True if cmp else False,
                          legendgroup='group', boxpoints=None if cmp else 'all'), row=2, col=1)

        fig.update_layout(legend_title_text=cmp,
                          boxmode='group' if cmp else None)

        if cmp:
            if cmp == 'dynamic':
                fig.update_layout(legend_title_text='Dynamic intensity C')
            elif cmp == 'problem':
                fig.update_layout(legend_title_text='Problem instance')

            txt = (
                f'Boxplot of best parameter sets (comparison of {cmp} from run {result_nums[0]} to {result_nums[-1]})')
        else:
            fig.update_traces(width=1/3)
            txt = (
                f'Boxplot of best parameter sets (run {result_nums[0]} to {result_nums[-1]})')

        fig.update_layout(boxgroupgap=0.2, boxgap=0.2)
        fig.update_layout(
            title={
                'text': txt,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        if self.output_path:
            fig.update_layout(title=None, font_family="Helvetica",
                              font_size=13, margin=dict(l=0, r=0, t=0, b=0))
            fig.write_image("/".join([self.output_path, f'parameter_boxplot_{cmp}.svg']),
                            format="svg", width=650 if cmp else 500, height=450)
            if export_values:
                if cmp:
                    results = results.groupby('cmp')
                with open("/".join([self.output_path, f'parameter_boxplot_stats_{cmp}_{result_nums[0]}_to_{result_nums[-1]}.csv']), 'w') as out:
                    out.write(results.quantile(
                        [0, 0.25, 0.5, 0.75, 1]).to_csv())

        else:
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

        if cmp == 'dynamic':
            results['cmp'] = results['cmp'].astype(str)

        fig = px.scatter_matrix(results,
                                dimensions=y,
                                color="cmp" if cmp else None)
        fig.update_traces(diagonal_visible=False, showupperhalf=False)

        if cmp:
            fig.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ))
            if cmp == 'dynamic':
                fig.update_layout(legend_title_text='Dynamic intensity C')
            elif cmp == 'problem':
                fig.update_layout(legend_title_text='Problem instance')

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

        if self.output_path:
            fig.update_layout(title=None, font_family="Helvetica",
                              margin=dict(l=0, r=0, t=0, b=0))
            fig.write_image("/".join([self.output_path, f'param_scatter_matrix_{cmp}.svg']),
                            format="svg", width=650, height=650)
        else:
            fig.show()

    def create_param_importance_plot(self, result_nums: list[int], cmp=None):
        results = pd.DataFrame()
        for n in result_nums:
            folder = self.load_result_folder(n)
            res = pd.DataFrame.from_dict(self.get_feature_importance(n))
            info = get_info(folder['info.json'])
            key = get_key_parameter(cmp, info)
            res['cmp'] = key
            results = pd.concat([results, res], sort=False)

        if cmp:
            mean = results.groupby('cmp', sort=False).mean().T
            error = results.groupby('cmp', sort=False).sem().T

            fig = go.Figure()
            for col in mean.columns:
                fig.add_trace(go.Bar(
                    name=col,
                    y=mean[col].index,
                    x=mean[col],
                    error_x=dict(type='data', array=[
                                 *error[col].to_dict().values()]),
                    orientation='h'
                ))

            fig.update_yaxes(autorange="reversed")

            txt = (
                f'Parameter Importance (compared by {cmp}, averaged over runs {result_nums[0]} to {result_nums[-1]})')
            fig.update_layout(barmode='group')

            if cmp == 'dynamic':
                fig.update_layout(legend_title_text='Dynamic intensity C')
            elif cmp == 'problem':
                fig.update_layout(legend_title_text='Problem instance')
        else:
            mean_results = pd.concat([results.mean(axis=0), results.sem(
                axis=0)], axis=1, keys=['value', 'error'])
            fig = px.bar(mean_results, x='value',
                         error_x='error', orientation='h')
            fig.update_traces(width=1)
            txt = (
                f'Parameter Importance (averaged over runs {result_nums[0]} to {result_nums[-1]})')

        if self.output_path:
            fig.update_layout(
                font_family="Helvetica",
                font_size=14,
                yaxis_title="Parameter",
                xaxis_title="Relative Importance (MDI)",
            )
            fig.update_layout(margin={'l': 0, 'r': 0, 't': 0, 'b': 0})
            fig.write_image("/".join([self.output_path, f'parameter_importance_bar_{cmp}.svg']),
                            format="svg", width=700 if cmp else 600, height=650 if cmp else 300)
        else:
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
            opt = get_optimizer_type(folder['info.json'])
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

    def create_partial_dep_plot(self, result_num: int, paths_dict=None, best_only=True, n_points=40):
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
        # remove reaction_type value, because expressiveness for two type categorical value is very low
        if 'reaction_type' in dimensions:
            dimensions.remove('reaction_type')

        index = self.get_best_param_set(result_num)[0]

        if best_only:
            plot_objective(results[index], dimensions=dimensions, plot_dims=[
                           0, 1, 2, 3, 4, 5], n_points=n_points)
            plt.suptitle('Partial Dependence (opt_%d, %s, %s [C=%.2f], run %d of %d)' %
                         (result_num, opt, problem, dynamic_intensity, index+1, len(results)))
        else:
            for i, res in enumerate(results):
                plot_objective(res, dimensions=dimensions, plot_dims=[
                    0, 1, 2, 3, 4, 5], n_points=n_points)
                plt.suptitle('Partial Dependence (opt_%d, %s, %s [C=%.2f], run %d of %d)' %
                             (result_num, opt, problem, dynamic_intensity, i+1, len(results)))

        if self.output_path:
            plt.suptitle(None)
            plt.savefig(
                "/".join([self.output_path, f'partial_dependence_{problem}_C_{dynamic_intensity}_run_{index+1}.svg']), dpi=300)
        else:
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
            conover_t, conover_p = sp.posthoc_conover(
                mean_min_val_list, p_adjust='bonf')
            conover_p.columns, conover_p.index = mean_min_key_list, mean_min_key_list
            conover_t.columns, conover_t.index = mean_min_key_list, mean_min_key_list

            tests['conover_t'] = conover_t.to_dict()
            tests['conover_p'] = conover_p.to_dict()

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

        if self.output_path:
            fig = make_subplots(shared_xaxes=True,
                                rows=1, cols=2, vertical_spacing=0.05, horizontal_spacing=0.05)
            fig1 = px.box(pd.DataFrame(func_results))
            fig2 = px.box(pd.DataFrame(auc_results))
            fig.add_trace(go.Box(fig1['data'][0]), row=1, col=1)
            fig.add_trace(go.Box(fig2['data'][0]), row=1, col=2)
            fig.update_layout(margin={'l': 0, 'r': 0, 't': 0, 'b': 0})
            fig.update_annotations(font_size=15)
            fig.write_image("/".join([self.output_path, 'convergence_stats_boxplot.svg']),
                            format="svg", width=650, height=250)
        else:
            fig = make_subplots(shared_xaxes=True,
                                rows=2, cols=1, vertical_spacing=0.05,
                                subplot_titles=("Relative difference to optimal solution quality<br>of all minimal function values ", "AUC of solution quality convergence graph"))
            fig1 = px.box(pd.DataFrame(func_results))
            fig2 = px.box(pd.DataFrame(auc_results))
            fig.add_trace(go.Box(fig1['data'][0]), row=1, col=1)
            fig.add_trace(go.Box(fig2['data'][0]), row=2, col=1)
            fig.update_layout(margin={'l': 10, 'r': 10, 't': 45, 'b': 10})
            fig.update_annotations(font_size=14)
            fig.show()

    def load_result_folder(self, result_num: int, results_path=None) -> Path:
        """
        Load the content of an output folder

        Args:
            result_num (int): Folder suffix to load, e.g. 5 for opt_5
            results_path (str, optional): Path to an alternative results folder, other than specified in the class

        Returns:
            Generator[Path, None, None]: Generator provided by Pathlib to access its contents 
        """

        if results_path is None:
            results_path = self.results_path
        path = "".join(
            [results_path, self.path_prefix, str(result_num), '/'])
        path_obj = Path(path)
        if path_obj.exists():
            files = {}
            for p in sorted(path_obj.iterdir()):
                if p.name == 'folder_info.json':
                    files['sub'] = True
                if p.is_dir():
                    files[p.name] = {x.name: x.as_posix()
                                     for x in Path(p.as_posix()).iterdir()}
                elif p.is_file():
                    files[p.name] = p.as_posix()
            return files
        return None

    def get_best_param_set(self, result_num: int) -> list[int, str, float, dict]:
        """
        Return the best param set found across all optimizer runs given the result number of a results folder.

        Args:
            result_num (int): Folder suffix to load, e.g. 5 for opt_5

        Returns:
            list[int, str, float, dict]: List of metrics for the best optimizer run: the iteration, instance name, dynamic intensity, and a dict of the specific parameters
        """
        folder = self.load_result_folder(result_num)
        params = load_opt_best_params(folder['opt_best_params.csv'], True)
        info = get_info(folder['info.json'])
        index = params['func_val'].idxmin()
        best_params = params.iloc[index]
        return (index, info['problem']['name'], info['problem']['dynamic_props']['dynamic_intensity'], best_params.to_dict())

    def get_best_param_cli(self, result_num: int) -> str:
        """
        Returns a command line interface version of the best parameter set found for a specific optimizer run, to use for the main.py CLI.

        Args:
            result_num (int): Folder suffix to load, e.g. 5 for opt_5

        Returns:
            str: CLI string for the main.py inputs
        """
        iteration, instance, dynamic_intensity, params = self.get_best_param_set(
            result_num)
        return f"-p {instance} -di {dynamic_intensity} -a {params['alpha']} -b {params['beta']} -plb {params['w_pers_best']:.3f} -ppr {params['w_pers_prev']:.3f} -ptb {params['w_parent_best']:.3f} -ddt {params['detection_threshold']:.3f} -r '{params['reaction_type']}'"

    def create_best_param_csv(self, result_nums: list[int]) -> None:
        """
        Saves a csv of the best parameter set found for a specific optimizer run.

        Args:
            result_num (int): Folder suffix to load, e.g. 5 for opt_5
        """
        csv = 'TSP,C,alpha,beta,w_pers_best,w_pers_prev,w_parent_best,detection_threshold,reaction_type\n'
        for i in result_nums:
            iteration, instance, dynamic_intensity, params = self.get_best_param_set(
                i)
            csv += ",".join([instance, str(dynamic_intensity),
                            *(list(map(str, params.values()))[:-1])])
            csv += '\n'

        with open("/".join([self.output_path, f'best_params_{result_nums[0]}_to_{result_nums[-1]}.csv']), 'w') as out:
            out.write(csv)

    def create_categorical_analysis_csv(self, result_nums: list[int]) -> None:
        results = pd.DataFrame()
        for n in result_nums:
            folder = self.load_result_folder(n)
            res = load_opt_best_params(
                folder['opt_best_params.csv'], func_val=True)
            info = get_info(folder['info.json'])
            res['dynamic'] = get_key_parameter('dynamic', info)
            res['problem'] = get_key_parameter('problem', info)

            opt_solution = get_optimal_solution(folder['info.json'])
            res['func_val'] = res['func_val'].apply(
                lambda x: (x-opt_solution)/opt_solution)

            results = pd.concat(
                [results, res], ignore_index=True, sort=False)

        csv = results['reaction_type'].value_counts(
            normalize=True).rename('all').to_frame().T

        res_dynamic = results.groupby('dynamic')
        temp = res_dynamic['reaction_type'].value_counts(normalize=True)
        csv = pd.concat(
            [csv, pd.concat({'dynamic_100': temp.unstack(level=1)})])

        for index, median in res_dynamic['func_val'].quantile(q=0.5).items():
            df = res_dynamic.get_group(index)
            temp = df[df['func_val'] > median]['reaction_type'].value_counts(
                normalize=True)
            temp = temp.reindex(['partial', 'full'])
            temp.name = '(dynamic_50, '+str(index)+')'
            csv = csv.append(temp)

        res_problem = results.groupby('problem')
        temp = res_problem['reaction_type'].value_counts(normalize=True)
        csv = pd.concat(
            [csv, pd.concat({'problem_100': temp.unstack(level=1)})])

        for index, median in res_problem['func_val'].quantile(q=0.5).items():
            df = res_problem.get_group(index)
            temp = df[df['func_val'] > median]['reaction_type'].value_counts(
                normalize=True)
            temp = temp.reindex(['partial', 'full'])
            temp.name = '(problem_50, '+str(index)+')'
            csv = csv.append(temp)

        with open("/".join([self.output_path, f'categorical_eval_{result_nums[0]}_to_{result_nums[-1]}.csv']), 'w') as out:
            out.write(csv.to_csv())

    def create_parameter_importance_csv(self, result_nums: list[int]) -> None:
        """
        Saves a csv of the parameter importance averaged over all provided results.
        Only possible for sklearn surrogate models, which provide this information.

        Args:
            result_num (int): Folder suffix to load, e.g. 5 for opt_5
        """
        results = pd.DataFrame()
        for n in result_nums:
            results = pd.concat([results, pd.DataFrame.from_dict(
                self.get_feature_importance(n))], sort=False)
        mean_results = pd.concat(
            [results.mean(axis=0), results.sem(axis=0)],
            axis=1, keys=['value', 'error']
        )

        with open("/".join([self.output_path, f'parameter_importance_{result_nums[0]}_to_{result_nums[-1]}.csv']), 'w') as out:
            out.write(mean_results.to_csv())

    def create_file_wrapper(self, result_num: int, filename: str, mode='a') -> None:
        """
        Initializes a file wrapper for logging

        Args:
            filename (str): Name of the file that should be created

        Returns:
            TextIOWrapper: Wrapper for the opened logging file
        """
        return open("".join(self.results_path, self.path_prefix, str(result_num), '/', filename), mode)


def load_opt_best_params(path: str, func_val=False) -> pd.DataFrame:
    """_summary_

    Args:
        path (str): _description_
        func_val (bool, optional): _description_. Defaults to False.

    Raises:
        FileExistsError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    try:
        df = pd.read_csv(path, sep=';')
        if func_val:
            df = df.loc[:, ~df.columns.isin(['run'])]
        else:
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
    elif cmp == 'detection_threshold':
        key = info['hsppbo']['detection_threshold']
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
    match info['optimizer']:
        case 'random':
            return 'RS'
        case 'bayesian':
            return 'GP'
        case 'forest':
            return 'ET'
        case 'gradient':
            return 'GBRT'
        case _:
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


def calc_reset_accuracy(row_data, trigger_points, positives, negatives) -> dict:
    r_pos = row_data.iloc[trigger_points]
    r_pos = r_pos.value_counts().to_dict()
    true_positives = r_pos.get(True, 0)

    r_neg = row_data.loc[row_data.index >= 2000]
    r_neg = r_neg[~r_neg.index.isin(trigger_points)]
    r_neg = r_neg.value_counts().to_dict()
    false_positives = r_neg.get(True, 0)

    fpr = false_positives / negatives
    recall = true_positives / positives
    try:
        precision = true_positives / (true_positives+false_positives)
    except ZeroDivisionError:
        precision = 0

    try:
        f1 = 2 * (recall*precision) / (recall+precision)
    except ZeroDivisionError:
        f1 = 0

    return {'p': positives, 'n': negatives, 'tp': true_positives, 'fp': false_positives, 'fpr': fpr, 'recall': recall, 'precision': precision, 'f1': f1}


def get_detection_info(info: dict) -> tuple[int, int, list[int]]:
    max_iter = info['hsppbo']['max_iteration_count']
    min_iter = info['problem']['dynamic_props']['min_iterations_before_dynamic']+1
    detection_pause = info['hsppbo']['detection_pause']
    dynamic_freq = info['problem']['dynamic_props']['dynamic_frequency']

    positives = int((max_iter - min_iter)/dynamic_freq)

    trigger_points = []

    for x in range(min_iter, max_iter):
        if x % dynamic_freq < detection_pause:
            trigger_points.append(x)

    negatives = max_iter - min_iter - len(trigger_points)

    return positives, negatives, trigger_points


def conv2siunitx(val, err):
    return (f'{val:.2f}({err:.2f})')


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
        diff = x_ranks_avg.loc[i] - x_ranks_avg.loc[j]
        B = (1. / x_lens.loc[i] + 1. / x_lens.loc[j])
        D = (n - 1. - h_cor) / (n - x_len)
        t_value = diff / np.sqrt(S2 * B * D)
        t_value_p = np.abs(diff) / np.sqrt(S2 * B * D)
        p_value = 2. * ss.t.sf(np.abs(t_value_p), df=n-x_len)
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

    h = (12. / (n * (n + 1.))) * \
        np.sum(x_ranks_sum**2 / x_lens) - 3. * (n + 1.)
    h_cor = h / x_ties

    if x_ties == 1:
        S2 = n * (n + 1.) / 12.
    else:
        S2 = (1. / (n - 1.)) * \
            (np.sum(x['ranks'] ** 2.) - (n * (((n + 1.)**2.) / 4.)))

    vs = np.zeros((x_len, x_len))
    tvs = np.zeros((x_len, x_len))
    tri_upper = np.triu_indices(vs.shape[0], 1)
    tri_lower = np.tril_indices(vs.shape[0], -1)
    vs[:, :] = 0
    tvs[:, :] = 0

    combs = it.combinations(range(x_len), 2)

    for i, j in combs:
        tvs[i, j], vs[i, j] = compare_conover(
            x_groups_unique[i], x_groups_unique[j])

    if p_adjust:
        vs[tri_upper] = multipletests(vs[tri_upper], method=p_adjust)[1]
    vs[tri_lower] = np.transpose(vs)[tri_lower]
    np.fill_diagonal(vs, 1)

    return pd.DataFrame(tvs, index=x_groups_unique, columns=x_groups_unique), pd.DataFrame(vs, index=x_groups_unique, columns=x_groups_unique)
