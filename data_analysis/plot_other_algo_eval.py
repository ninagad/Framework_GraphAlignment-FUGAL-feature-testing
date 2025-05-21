import math
import os
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.figure import Figure

from utils import get_acc_file_as_df, get_algo_args, get_graph_names_from_file, strip_graph_name, get_git_root


allowed_colormaps = Literal['Greens', 'Blues']
def compute_mean_over_iters(source: int):
    df = get_acc_file_as_df(source)
    df = df.replace(-1, np.nan)  # Replace numeric errors with NaN, so they are excluded from the mean calculation.
    df['mean'] = df.mean(axis=1)

    return df


def get_color_map(traces: int, hue: allowed_colormaps):
    # Choose a colormap and generate colors for the lines
    cmap = plt.get_cmap(hue)
    colors = [cmap(x) for x in np.linspace(0.4, 0.9, traces)]
    return colors


def get_marks():
    marker_options = ['o', '^', 's', 'D', 'x', 'P', 'd']
    return marker_options


def plot_cone_subplots(df: pd.DataFrame, source: int, axes, row: int, col: int, hue: allowed_colormaps):
    # Divide df into each run
    dfs = [df.iloc[i:i + 6] for i in range(0, len(df), 6)]

    colors = get_color_map(len(dfs), hue)
    marks = get_marks()

    xs = df.index.get_level_values(1).unique()

    args_lst = get_algo_args(source)
    for idx, (args, df) in enumerate(zip(args_lst, dfs)):
        dist_scalar = args['dist_scalar']

        # Format to scientific notation
        exponent = int(math.log10(dist_scalar))

        axes[row, col].plot(100*xs, 100*df['mean'], label=f'$10^{{{exponent}}}$', color=colors[idx], marker=marks[idx],
                            markersize=5)


def plot_subplot(baseline: int, source: int, axes, row: int, col: int, title: str, additional_trace: int | None):
    #baseline_color = '#2596be'
    baseline_color = '#8eb576'
    # Green pair
    baseline_color = '#b1de89'
    color = '#31a354'

    #color = '#688557'
    #color = '#38b1d9'

    #baseline_color = '#4daf4a'
    #color = '#377eb8'

    graph_name = get_graph_names_from_file([source])[0]
    graph_name = strip_graph_name(graph_name)

    df = get_acc_file_as_df(source)
    df = df.replace(-1, np.nan)

    baseline_df = get_acc_file_as_df(baseline)
    baseline_df = baseline_df.replace(-1, np.nan)

    # Get noise levels
    xs = df.index.get_level_values(1).unique()

    if "CONE" in title:
        df = compute_mean_over_iters(source)
        baseline_df = compute_mean_over_iters(baseline)
        if 'alignment' in title:
            hue: allowed_colormaps = 'Blues'
        else:
            hue: allowed_colormaps = 'Greens'

        plot_cone_subplots(df, source, axes, row, col, hue)

        if 'alignment' in title:
            lr_color = plt.get_cmap("tab10")(3)
            lr_df = compute_mean_over_iters(additional_trace)
            axes[row, col].plot(100 * xs, 100 * lr_df['mean'], label='lr = 0', color=lr_color)

        baseline_color = plt.get_cmap("tab10")(1)
        axes[row, col].plot(100*xs, 100*baseline_df['mean'], label='baseline', color=baseline_color)

        axes[row, col].grid(True)
        axes[row, col].set_ylim(-10, 110)
    else:
        baseline_df['type'] = 'baseline'
        df['type'] = 'with features'
        baseline_df['noise'] = xs
        df['noise'] = xs

        # Combine into one DataFrame
        df_all = pd.concat([baseline_df, df], ignore_index=True)

        # Collapse iteration columns into a single column of accuracies
        df_all = df_all.melt(
            id_vars=['type', 'noise'],
            var_name='iteration',
            value_name='accuracy')

        df_all['accuracy'] = 100 * df_all['accuracy']
        df_all['noise'] = (100 * df_all['noise']).astype(int)

        sns.barplot(data=df_all,
                    x='noise',
                    y='accuracy',
                    hue='type',
                    ax=axes[row, col],
                    palette=[baseline_color, color],
                    edgecolor="black",
                    linewidth=0.6,
                    errorbar="sd"
                    )
        # make the background grid visible
        axes[row, col].grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.7)
        axes[row, col].set_axisbelow(True)  # keep bars in front of the grid

        # hide the individual legend
        axes[row, col].legend().remove()

    # Layout plot
    if col == 0:
        axes[row, col].set_ylabel('Avg. accuracy (%)')
    axes[row, col].set_xlabel('Noise level (%)')
    axes[row, col].set_title(label=f'{graph_name}', fontsize=12)


def layout_plot(fig: Figure, axes, title: str, legend_name: str):
    if "CONE" in title:
        plt.tight_layout(rect=(0, 0, 0.87, 0.95))  # restrict tight_layout to the reduced area
    else:
        plt.tight_layout(rect=(0, 0, 0.83, 0.95))

    # Add legend outside the plot, top-right
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper right',
               bbox_to_anchor=(1.0, 0.908),
               title=legend_name)

    # Center suptitle w.r.t. plot areas excluding the tick and axis labels.
    fig.canvas.draw()  # required to update layout info
    pos = [ax.get_position() for ax in axes[0]]
    center_x = (pos[0].x0 + pos[-1].x1) / 2

    plt.suptitle(title, x=center_x, fontsize=18)


def plot_eval_graphs(baselines: list, sources: list, title: str, legend_title: str, filename: str,
                     additional_trace=None):
    if additional_trace is None:
        additional_trace = [None, None, None, None]
    else:
        assert len(additional_trace) == 4

    assert len(baselines) == 4
    assert len(sources) == 4

    # Create one figure with a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2,
                             ncols=2,
                             figsize=(8, 7),
                             sharey='row')

    for i, (baseline, source, additional) in enumerate(zip(baselines, sources, additional_trace)):
        row = i // 2
        col = i % 2

        plot_subplot(baseline, source, axes, row, col, title, additional)

    layout_plot(fig, axes, title, legend_title)

    # Save plot
    root_path = get_git_root()
    sub_dir = title.split()[0]
    path = os.path.join(root_path, 'plots', 'Other-algorithms', sub_dir, f'{filename}.pdf')
    plt.savefig(path)


if __name__ == '__main__':
    # CONE convex initialization
    cone_baselines = [281, 282, 283, 284]
    cone_sources = [14054, 14179, 14181, 14182]
    plot_eval_graphs(cone_baselines, cone_sources, 'CONE - convex initialization', 'Dist scalar', 'CONE-convex-init-dist_scalar')

    # CONE optimal matching
    cone_sources = [14463, 14465, 14467, 14468]
    cone_lr_0 = [15161, 15163, 15164, 15165]
    plot_eval_graphs(cone_baselines,cone_sources, 'CONE - alignment', 'Dist scalar', 'CONE-alignment-dist_scalar', cone_lr_0)

    isorank_baselines = [15274, 15276, 15279, 15280]
    isorank_sources = [15272, 15258, 15260, 15269]
    plot_eval_graphs(isorank_baselines, isorank_sources, 'IsoRank', '', 'IsoRank-bar-eval')

    regal_baselines = [15211, 15196, 15215, 15201]
    regal_sources = [15126, 15129, 15142, 15123]
    plot_eval_graphs(regal_baselines, regal_sources, 'REGAL', '', 'REGAL-bar-eval')

    grampa_baselines = [15285, 15287, 15289, 15291]
    grampa_sources = [15233, 15235, 15239, 15240]
    plot_eval_graphs(grampa_baselines, grampa_sources, 'GRAMPA', '', 'GRAMPA-bar-eval')
