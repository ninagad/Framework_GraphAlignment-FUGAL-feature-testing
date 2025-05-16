import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.figure import Figure

from utils import get_acc_file_as_df, get_algo_args, get_graph_names_from_file, strip_graph_name, get_git_root


def compute_mean_over_iters(source: int):
    df = get_acc_file_as_df(source)
    df = df.replace(-1, np.nan)  # Replace numeric errors with NaN, so they are excluded from the mean calculation.
    df['mean'] = df.mean(axis=1)

    return df


def get_color_map(traces: int):
    # Choose a colormap and generate colors for the lines
    cmap = plt.get_cmap("Greens")
    colors = [cmap(x) for x in np.linspace(0.3, 0.9, traces)]
    return colors


def get_marks():
    marker_options = ['o', '^', 's', 'D', 'x', 'P', 'd']
    return marker_options


def plot_cone_subplots(df: pd.DataFrame, source: int, axes, row: int, col: int):
    # Divide df into each run
    dfs = [df.iloc[i:i + 6] for i in range(0, len(df), 6)]

    colors = get_color_map(len(dfs))
    marks = get_marks()

    xs = df.index.get_level_values(1).unique()

    args_lst = get_algo_args(source)
    for idx, (args, df) in enumerate(zip(args_lst, dfs)):
        dist_scalar = args['dist_scalar']

        axes[row, col].plot(xs, df['mean'], label=dist_scalar, color=colors[idx], marker=marks[idx],
                            markersize=5)


def plot_subplot(baseline: int, source: int, axes, row: int, col: int, title: str):
    # Baseline trace color (orange)
    #baseline_color = plt.get_cmap("Set1")(4)
    colormap = plt.get_cmap('tab10')
    baseline_color = colormap(1)
    color = colormap(0)

    graph_name = get_graph_names_from_file([source])[0]
    graph_name = strip_graph_name(graph_name)

    df = compute_mean_over_iters(source)

    # Get noise levels
    xs = df.index.get_level_values(1).unique()

    # Baseline
    baseline_df = compute_mean_over_iters(baseline)

    if title == "CONE":
        axes[row, col].plot(xs, baseline_df['mean'], label='baseline', color=baseline_color)
        plot_cone_subplots(df, source, axes, row, col)
        axes[row, col].grid(True)
        axes[row, col].set_ylim(-0.1, 1.1)
    else:
        baseline_df['type'] = 'baseline'
        df['type'] = 'with features'
        baseline_df['noise'] = xs
        df['noise'] = xs
        # Combine into one DataFrame
        df_all = pd.concat([baseline_df, df], ignore_index=True)
        sns.barplot(data=df_all, x='noise', y='mean', hue='type', ax=axes[row, col], palette=[baseline_color, color])
        # hide the individual legend
        axes[row, col].legend().remove()

    # Layout plot
    if col == 0:
        axes[row, col].set_ylabel('Accuracy')
    axes[row, col].set_xlabel('Noise level')
    axes[row, col].set_title(label=f'{graph_name}', fontsize=12)


def layout_plot(fig: Figure, axes, title: str, legend_name: str):
    if title == 'CONE':
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


def plot_eval_graphs(baselines: list, sources: list, title: str, legend_title: str, filename: str):
    assert len(baselines) == 4
    assert len(sources) == 4

    # Create one figure with a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2,
                             ncols=2,
                             figsize=(8, 7),
                             sharey='row')

    for i, (baseline, source) in enumerate(zip(baselines, sources)):
        row = i // 2
        col = i % 2

        plot_subplot(baseline, source, axes, row, col, title)

    layout_plot(fig, axes, title, legend_title)

    # Save plot
    root_path = get_git_root()
    path = os.path.join(root_path, 'plots', 'Other-algorithms', title, f'{filename}.pdf')
    plt.savefig(path)


if __name__ == '__main__':
    cone_baselines = [281, 282, 283, 284]
    cone_sources = [14054, 14179, 14181, 14182]
    plot_eval_graphs(cone_baselines, cone_sources, 'CONE', 'Dist scalar', 'CONE-dist_scalar')

    isorank_baselines = [15274, 15276, 15279, 15280]
    isorank_sources = [15272, 15258, 15260, 15269]
    plot_eval_graphs(isorank_baselines, isorank_sources, 'IsoRank', '', 'IsoRank-bar-eval')

    regal_baselines = [15211, 15196, 15215, 15201]
    regal_sources = [15126, 15129, 15142, 15123]
    plot_eval_graphs(regal_baselines, regal_sources, 'REGAL', '', 'REGAL-bar-eval')

    grampa_baselines = [15285, 15287, 15289, 15291]
    grampa_sources = [15233, 15235, 15239, 15240]
    plot_eval_graphs(grampa_baselines, grampa_sources, 'GRAMPA', '', 'GRAMPA-bar-eval')
