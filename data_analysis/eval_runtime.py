import os
from typing import Literal
from enum import Enum

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_analysis.utils import get_total_time_as_df, get_graph_names_from_file, get_git_root, eval_bar_plot_palette
from run_tests import save_config_info
from data_analysis.utils import AlgorithmEnums


def load_data(sources: dict[AlgorithmEnums, list[int]]):
    dfs = []
    for alg, runs in sources.items():
        for run in runs:
            run_df = get_total_time_as_df(run)
            graph = get_graph_names_from_file([run], 'Time-runs')[0]
            run_df['graph'] = graph
            run_df['algorithm'] = alg.value
            dfs.append(run_df)

    combined_df = pd.concat(dfs)

    # Collapse iterations into a single column
    combined_df = combined_df.reset_index(level=['Noise'])

    combined_df = combined_df.melt(
        id_vars=['algorithm', 'Noise', 'graph'],
        var_name='iteration',
        value_name='runtime')

    return combined_df


def save_df(df: pd.DataFrame):
    root = get_git_root()
    path = os.path.join(root, 'plot-data', 'runtime-plot.txt')

    df['mean'] = df.groupby(['algorithm', 'graph'])['runtime'].transform('mean')

    # Configure options to save all rows and columns in save file
    pd.set_option('display.max_rows', len(df))
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)

    save_config_info(path, df.sort_values(['graph', 'algorithm', 'Noise', 'iteration']).reset_index(drop=True))


def plot_subplot(subplot: plt.Axes, data: pd.DataFrame):
    palette = eval_bar_plot_palette()

    sns.barplot(data=data,
                x='graph',
                y='runtime',
                hue='algorithm',
                ax=subplot,
                palette=palette,
                edgecolor="black",
                linewidth=0.6,
                errorbar=None,
                width=0.6  # Default is 0.8; lower values make bars thinner
                )

    subplot.set_xlabel('Graph', fontsize=14)
    subplot.set_ylabel('Time (sec)', fontsize=14)

    # make the background grid visible
    subplot.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=1)
    subplot.set_axisbelow(True)  # keep bars in front of the grid

    # Change fontsize of tick labels
    subplot.tick_params(axis='both', labelsize=14)

    # hide the individual legend
    subplot.legend().remove()

    if data['runtime'].max() > 400:
        subplot.set_ylim(0, 505)


def plot_runtime(data: pd.DataFrame):
    df_small = data[data['graph'].isin(['ia-crime-moreno', 'power-685-bus', 'DD_g501'])]
    df_large = data[data['graph'].isin(['inf-power', 'bio-yeast', 'socfb-Bowdoin47'])]

    # Create one figure with a grid of subplots
    fig, axes = plt.subplots(nrows=1,
                             ncols=2,
                             figsize=(11.5, 4.5)
                             )
    plot_subplot(axes[0], df_small)
    plot_subplot(axes[1], df_large)

    plt.tight_layout(rect=(0, 0, 1, 0.81))  # restrict tight_layout to the reduced area

    # Create legend
    handles, labels = axes[0].get_legend_handles_labels()
    pos = [ax.get_position() for ax in axes]
    center_x = (pos[0].x0 + pos[-1].x1) / 2

    legend_x = pos[-1].x1 + 0.01
    legend_y = pos[-1].y1 + 0.25

    fig.legend(handles, labels,
               loc='upper right',
               bbox_to_anchor=(legend_x, legend_y),
               bbox_transform=fig.transFigure,
               fontsize=14
               )
    plt.suptitle('Runtimes of FUGAL variants', x=center_x, fontsize=22)

    root_path = get_git_root()
    path = os.path.join(root_path, 'plots', 'FUGAL-evaluation', 'runtime-plot.pdf')
    plt.savefig(path)


if __name__ == '__main__':
    source_dict = {AlgorithmEnums.ORIGINAL_FUGAL: [17241, 17244, 17247, 17250, 17253, 17256],
                   AlgorithmEnums.FUGAL_FIXED: [17239, 17242, 17245, 17248, 17251, 17254],
                   AlgorithmEnums.FUGAL_PCA: [17240, 17243, 17246, 17249, 17252, 17255]}

    df = load_data(source_dict)
    #save_df(df)
    plot_runtime(df)
