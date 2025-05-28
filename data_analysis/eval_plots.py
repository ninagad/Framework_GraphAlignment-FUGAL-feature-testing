import math
import os
from typing import Literal

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.figure import Figure

from data_analysis.utils import get_acc_file_as_df, get_algo_args, get_graph_names_from_file, strip_graph_name, \
    get_git_root
from data_analysis.test_run_configurations import test_graph_set_are_equal

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


def plot_cone_subplots(df: pd.DataFrame, source: int, subplot, hue: allowed_colormaps):
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

        subplot.plot(100 * xs, 100 * df['mean'], label=f'$10^{{{exponent}}}$', color=colors[idx],
                     marker=marks[idx],
                     markersize=5)


def compute_confidence_interval(x):
    # unique values
    unique = np.unique(x)

    # Return the lower and upper bound on the confidence interval, if there is more than one unique value.
    # Otherwise, return the unique value as both lower and upper bound.
    if len(unique) > 1:
        lower, upper = stats.t.interval(confidence=0.95, df=len(x) - 1, loc=np.mean(x),
                                                scale=np.std(x, ddof=1) / np.sqrt(len(x)))
        return float(lower), float(upper)
    else:
        return None, None


def plot_subplot(baseline: int, source: int, subplot, col: int, title: str, additional_trace: int | None):
    # Check that baseline and source is computed on the same graph
    test_graph_set_are_equal(baseline, source)

    # baseline_color = '#2596be'
    baseline_color = '#8eb576'
    # Green pair
    baseline_color = '#b1de89'
    color = '#31a354'

    # color = '#688557'
    # color = '#38b1d9'

    # baseline_color = '#4daf4a'
    # color = '#377eb8'

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

        plot_cone_subplots(df, source, subplot, hue)

        if 'alignment' in title:
            lr_color = plt.get_cmap("tab10")(3)
            lr_df = compute_mean_over_iters(additional_trace)
            subplot.plot(100 * xs, 100 * lr_df['mean'], label='lr = 0', color=lr_color)

        baseline_color = plt.get_cmap("tab10")(1)
        subplot.plot(100 * xs, 100 * baseline_df['mean'], label='Original', color=baseline_color)

        subplot.grid(True)
        subplot.set_ylim(-10, 110)
    else:
        baseline_df['type'] = 'Original'
        df['type'] = 'Proposed'
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
                    ax=subplot,
                    palette=[baseline_color, color],
                    edgecolor="black",
                    linewidth=0.6,
                    # errorbar="sd"
                    errorbar=lambda x: compute_confidence_interval(x),
                    err_kws={"linewidth": 1},
                    capsize=0.2,
                    )
        # make the background grid visible
        subplot.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=1)
        subplot.set_axisbelow(True)  # keep bars in front of the grid

        # hide the individual legend
        subplot.legend().remove()
        subplot.set_ylim(0, 110)

    # Layout plot
    if col == 0:
        subplot.set_ylabel('Avg. accuracy (%)')
    subplot.set_xlabel('Noise level (%)')
    subplot.set_title(label=f'{graph_name}', fontsize=12)


def layout_plot(fig: Figure, axes, title: str, legend_name: str):
    if "CONE" in title:
        plt.tight_layout(rect=(0, 0, 0.85, 0.95))  # restrict tight_layout to the reduced area
    else:
        plt.tight_layout(rect=(0, 0, 1, 0.89))

    # Add legend outside the plot, top-right
    try:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        pos = [ax.get_position() for ax in axes[0]]
        center_x = (pos[0].x0 + pos[-1].x1) / 2
        if 'CONE' in title:
            legend_x = pos[-1].x1 + 0.175
            legend_y = pos[-1].y1 + 0.01
        else:
            legend_x = pos[-1].x1 + 0.01
            legend_y = pos[-1].y1 + 0.13

    except TypeError:  # only one plot
        handles, labels = axes.get_legend_handles_labels()
        pos = axes.get_position()
        center_x = (pos.x0 + pos.x1) / 2
        legend_x = pos.x1 + 0.01
        legend_y = pos.y1 + 0.23

    fig.legend(handles, labels,
               loc='upper right',
               bbox_to_anchor=(legend_x, legend_y),
               bbox_transform=fig.transFigure,
               title=legend_name)

    # Center suptitle w.r.t. plot areas excluding the tick and axis labels.
    fig.canvas.draw()  # required to update layout info

    plt.suptitle(title, x=center_x, fontsize=18)


def plot_eval_graphs(baselines: list, sources: list, title: str, legend_title: str = '',
                     additional_trace=None):
    if additional_trace is None:
        additional_trace = len(baselines) * [None]

    assert len(baselines) == len(sources)
    assert len(baselines) == len(additional_trace)

    rows = 2 if len(baselines) != 1 else 1
    cols = math.ceil(len(baselines) / rows)
    # Create one figure with a grid of subplots
    fig, axes = plt.subplots(nrows=rows,
                             ncols=cols,
                             figsize=(3.5 * cols, 3.5 * rows),
                             sharey='row')

    for i, (baseline, source, additional) in enumerate(zip(baselines, sources, additional_trace)):
        row = i // cols
        col = i % cols

        subplot = axes[row, col] if len(baselines) != 1 else axes
        plot_subplot(baseline, source, subplot, col, title, additional)

    layout_plot(fig, axes, title, legend_title)

    return fig


def save_fig(fig: plt.Figure, filename: str, subdir: str = ''):
    # Save plot
    root_path = get_git_root()
    path = os.path.join(root_path, 'plots', subdir, f'{filename}.pdf')
    fig.savefig(path)


def other_algo_eval():
    # CONE convex initialization
    subdir = os.path.join('Other-algorithms', 'CONE')
    cone_baselines = [281, 282, 283, 284]
    cone_sources = [14054, 14179, 14181, 14182]
    fig = plot_eval_graphs(cone_baselines, cone_sources, 'CONE - convex initialization', 'Dist scalar')
    save_fig(fig, 'CONE-convex-init-dist_scalar', subdir)

    # CONE optimal matching
    cone_sources = [14463, 14465, 14467, 14468]
    cone_lr_0 = [15161, 15163, 15164, 15165]
    fig = plot_eval_graphs(cone_baselines, cone_sources, 'CONE - alignment', 'Dist scalar',
                           cone_lr_0)
    save_fig(fig, 'CONE-alignment-dist_scalar', subdir)

    # IsoRank
    subdir = os.path.join('Other-algorithms', 'IsoRank')

    # Without degree similarity
    # isorank_baselines = [15274, 15276, 15279, 15280]

    # With degree similarity
    isorank_baselines = [16098, 16095, 16096, 16097] # 17268, 17269, 17270, 17274] # crime-moreno, inf-power, bio-yeast, mahindas
    isorank_sources = [15272, 15258, 15260, 15269] # 17271, 17272, 17273, 17275] # crime-moreno, inf-power, bio-yeast, mahindas
    fig = plot_eval_graphs(isorank_baselines, isorank_sources, 'IsoRank')
    save_fig(fig, 'IsoRank-bar-eval', subdir)

    # REGAL
    subdir = os.path.join('Other-algorithms', 'REGAL')
    regal_baselines = [15211, 15196, 15215, 15201] # 17295, 17290, 17289, 17288] # crime-moreno, inf-power, bio-yeast, mahindas
    regal_sources = [15126, 15129, 15142, 15123] # 17294, 17291, 17292, 17293] # crime-moreno, inf-power, bio-yeast, mahindas
    fig = plot_eval_graphs(regal_baselines, regal_sources, 'REGAL')
    save_fig(fig, 'REGAL-bar-eval', subdir)

    # GRAMPA
    subdir = os.path.join('Other-algorithms', 'GRAMPA')
    grampa_baselines = [15285, 15287, 15289, 15291] # 17280, 17281, 17282, 17283] # crime-moreno, inf-power, bio-yeast, mahindas
    grampa_sources = [15233, 15235, 15239, 15240] # 17284, 17287, 17286, 17285] # crime-moreno, inf-power, bio-yeast, mahindas
    fig = plot_eval_graphs(grampa_baselines, grampa_sources, 'GRAMPA')
    save_fig(fig, 'GRAMPA-bar-eval', subdir)


def fugal_eval():
    # inf-power, crime, bus,
    # facebook 47, bio-yeast, dd
    baselines = [17247, 17245, 17243, 17241, 17239, 17242]
    sources = [17236, 16121, 16088, 16085, 17235, 16086]
    fig = plot_eval_graphs(baselines, sources, 'FUGAL')
    save_fig(fig, 'primary-eval', 'FUGAL-evaluation')

    econ_baseline = [16385]  # econ-mahindas
    econ_source = [16374]  # econ-mahindas
    fig = plot_eval_graphs(econ_baseline, econ_source, '')
    save_fig(fig, 'econ-eval', 'FUGAL-evaluation')

    # email-univ, in-arenas, dublin, tomography
    appendix_baselines = [16388, 17253, 17244, 17264]
    appendix_sources = [16377, 17265, 16087, 17266]
    fig = plot_eval_graphs(appendix_baselines, appendix_sources, 'FUGAL')
    save_fig(fig, 'appendix-eval', 'FUGAL-evaluation')


if __name__ == '__main__':
    fugal_eval()
    other_algo_eval()
