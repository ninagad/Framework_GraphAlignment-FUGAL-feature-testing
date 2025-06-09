import os.path
from enum import Enum
from typing import Literal

import pandas as pd
import matplotlib.pyplot as plt

from data_analysis.utils import AlgorithmEnums, get_git_root, get_graph_names_from_file, get_acc_file_as_df, get_metric, \
    get_noise_type, eval_bar_plot_palette


def load_data(sources: dict[AlgorithmEnums, list[int]]) -> pd.DataFrame:
    dfs = []
    for algorithm, runs in sources.items():

        for run in runs:
            df = get_acc_file_as_df(run)

            df['mean'] = df.mean(axis=1)

            metric = get_metric(run)
            noise_type = get_noise_type(run)

            df['metric'] = metric
            df['noise-type'] = noise_type
            df['algorithm'] = algorithm.value

            # Convert Features and noises to columns
            df = df.reset_index()
            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df


def plot_subplot(df: pd.DataFrame, subplot: plt.Axes):
    palette = eval_bar_plot_palette()
    markers = ['o', 's', 'D']
    algorithms = [AlgorithmEnums.ORIGINAL_FUGAL, AlgorithmEnums.FUGAL_FIXED, AlgorithmEnums.FUGAL_PCA]

    for color, marker, alg in zip(palette, markers, algorithms):
        alg_df = df[df['algorithm'] == alg.value]

        subplot.plot(alg_df['Noise'], alg_df['mean'], label=alg.value, color=color, marker=marker)

    subplot.grid(True)


def check_graph_config(expected_graph: str, runs: list[int]):
    graphs = get_graph_names_from_file(runs)
    if (len(set(graphs)) != 1) or graphs[0] != expected_graph:
        raise ValueError(f'Graphs should be {expected_graph}, but is {graphs}')


def plot_noise_and_metrics_eval(df: pd.DataFrame, graph: str):
    noise_types = ['One-way', 'Multi-modal', 'Two-way']
    metrics = ['$S^3$', 'MNC', 'Frobenius norm']

    rows = 3
    cols = 3
    fig, axes = plt.subplots(nrows=rows,
                             ncols=cols,
                             figsize=(3 * 3, 2.5 * 3),
                             sharey='row',
                             sharex='col')

    for row in range(rows):
        for col in range(cols):
            subplot = axes[row, col]

            metric = metrics[row]
            noise_type = noise_types[col]

            metric_noise_data = df[(df['noise-type'] == noise_type) & (df['metric'] == metric)]

            if (metric == 'MNC') or (metric == '$S^3$'):
                metric_noise_data['mean'] = 100 * metric_noise_data['mean']
                subplot.set_ylim(-1, 110)

            plot_subplot(metric_noise_data, subplot)

    # Common y-labels
    for row in range(rows):
        axes[row, 0].set_ylabel(metrics[row], fontsize=12)

    # Common x-labels
    for col in range(cols):
        axes[2, col].set_xlabel(f'{noise_types[col]} noise (%)', fontsize=12)

    plt.tight_layout(rect=(0, 0, 1, 0.92))

    # Get handles and labels from one of the subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Add a single global legend
    fig.legend(handles, labels,
               loc='upper right',
               bbox_to_anchor=(0.99, 1.005),
               bbox_transform=fig.transFigure, )

    root = get_git_root()
    path = os.path.join(root, 'plots', 'FUGAL-evaluation', f'Other-metrics-and-noises-{graph}.pdf')
    plt.savefig(path)


if __name__ == '__main__':
    # ia-crime-moreno
    # mnc_1, s3_1, frob_1, mnc_2, s3_2, frob_2, mnc_3, s3_3, frob_3
    ia_crime_sources = {AlgorithmEnums.ORIGINAL_FUGAL: [22404, 22403, 22405, 23394, 23388, 23399, 23396, 23390, 23400],
                        AlgorithmEnums.FUGAL_FIXED:  [23369, 23366, 23372, 23385, 23383, 23387, 23386, 23384, 23389],
                        AlgorithmEnums.FUGAL_PCA: [23402, 23401, 23403, 23395, 23392, 23398, 23393, 23391, 23397],
                        }

    ia_graph = 'ia-crime-moreno'

    sources_lst = [(ia_graph, ia_crime_sources), ]

    for graph_name, sources in sources_lst:
        data = load_data(sources)
        check_graph_config(graph_name,
                           sources[AlgorithmEnums.ORIGINAL_FUGAL]
                           + sources[AlgorithmEnums.FUGAL_FIXED])
        check_graph_config(graph_name, sources[AlgorithmEnums.FUGAL_PCA])

        plot_noise_and_metrics_eval(data, graph_name)
