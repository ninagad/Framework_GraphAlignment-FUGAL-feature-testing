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
    get_git_root, eval_bar_plot_palette, AlgorithmEnums
from data_analysis.test_run_configurations import test_graph_set_are_equal
from scripts.run_utils import save_config_info

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
    marker_options = ['o', 's', 'D', '^', 'x', 'P', 'd', '*']
    return marker_options


def plot_cone_subplots(df: pd.DataFrame, source: int, subplot: plt.Axes, hue: allowed_colormaps):
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


def save_data(df, path, graph):
    save_df = df.copy()

    # Save plot data to file
    for noise in df['noise'].unique():
        for algo in df['type'].unique():
            accs = df.groupby(['type', 'noise']).get_group((algo, noise))['accuracy'].values
            lower, upper = compute_confidence_interval(accs)
            condition = (save_df['noise'] == noise) & (save_df['type'] == algo)
            save_df.loc[condition, 'ci_lower'] = lower
            save_df.loc[condition, 'ci_upper'] = upper

    save_df['avg acc'] = save_df.groupby(['noise', 'type'])['accuracy'].transform('mean')

    # Configure options to save all rows and columns in save file
    pd.set_option('display.max_rows', len(save_df))
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)

    root = get_git_root()
    path = os.path.join(root, 'plot-data', path)
    save_config_info(path, f'\n{graph}')
    save_config_info(path, save_df.sort_values(['type', 'noise']))


def format_CONE_plot(source_id: int, traces: dict, xs: list[float], title: str, subplot: plt.Axes):
    for label, trace_df in traces.items():
        trace_df['mean'] = trace_df.mean(axis=1)

    if 'Alignment' in title:
        hue: allowed_colormaps = 'Blues'
    else:
        hue: allowed_colormaps = 'Greens'

    plot_cone_subplots(traces['Proposed'], source_id, subplot, hue)

    if 'Alignment' in title:
        lr_df = traces['lr = 0']
        lr_color = plt.get_cmap("tab10")(3)
        subplot.plot(100 * xs, 100 * lr_df['mean'], label='lr = 0', color=lr_color)

    baseline = traces['Original']
    baseline_color = plt.get_cmap("tab10")(1)
    subplot.plot(100 * xs, 100 * baseline['mean'], label='Original', color=baseline_color)

    subplot.grid(True)
    subplot.set_ylim(-1, 110)


def format_barplot(traces: dict, subplot: plt.Axes, xs: list[float], graph: str, data_save_path: str):
    palette = eval_bar_plot_palette()

    for algorithm, df in traces.items():
        df['type'] = algorithm.value
        df['noise'] = xs

    # Combine into one DataFrame
    df_all = pd.concat(list(traces.values()), ignore_index=True)

    # Collapse iteration columns into a single column of accuracies
    df_all = df_all.melt(
        id_vars=['type', 'noise'],
        var_name='iteration',
        value_name='accuracy')

    df_all['accuracy'] = 100 * df_all['accuracy']
    df_all['noise'] = (100 * df_all['noise']).astype(int)

    #save_data(df_all, data_save_path, graph)

    sns.barplot(data=df_all,
                x='noise',
                y='accuracy',
                hue='type',
                ax=subplot,
                palette=palette,
                edgecolor="black",
                linewidth=0.6,
                # errorbar="sd"
                errorbar=lambda x: compute_confidence_interval(x),
                err_kws={"linewidth": 1, 'color': 'black'},
                capsize=0.2,
                )

    # make the background grid visible
    subplot.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=1)
    subplot.set_axisbelow(True)  # keep bars in front of the grid

    # hide the individual legend
    subplot.legend().remove()
    subplot.set_ylim(0, 110)


def format_line_plot(traces: dict[AlgorithmEnums, pd.DataFrame], xs: list[float], subplot: plt.Axes):
    marks = get_marks()
    cmap = plt.get_cmap('tab20')
    formatting = {#AlgorithmEnums.ORIGINAL_FUGAL: (marks[0], cmap(1)),  # Orange
                  AlgorithmEnums.FUGAL_PCA: (marks[1], cmap(6)),  # Red
                  AlgorithmEnums.GRAMPA_FIXED: (marks[2], cmap(0)),  # Blue
                  AlgorithmEnums.ISORANK_PCA: (marks[3], cmap(4)),  # Green
                  AlgorithmEnums.REGAL_PCA: (marks[4], cmap(8)),  # Purple

                  AlgorithmEnums.ORIGINAL_FUGAL: (marks[5], cmap(6)),
                  AlgorithmEnums.ORIGINAL_GRAMPA: (marks[6], cmap(0)),
                  AlgorithmEnums.ORIGINAL_ISORANK: (marks[7], cmap(4)),
                  AlgorithmEnums.ORIGINAL_REGAL: (marks[0], cmap(8))
                  }

    for algorithm, df in traces.items():
        df['mean'] = 100 * df.mean(axis=1)

        mark, color = formatting[algorithm]
        subplot.plot(100 * xs, df['mean'], label=algorithm.value, color=color, marker=mark)

    subplot.grid(True)
    subplot.set_ylim(-1, 110)


def plot_subplot(traces: dict, subplot: plt.Axes, col: int, title: str, data_path: str):
    # Check that baseline and source is computed on the same graph
    # TODO: add check back in
    # test_graph_set_are_equal(baseline, source)

    graph_name = get_graph_names_from_file([list(traces.values())[0]])[0]
    graph_name = strip_graph_name(graph_name)
    dfs = {}
    for label, source in traces.items():
        df = get_acc_file_as_df(source)
        if -1 in df.values:
            raise ValueError(f'Run {source} has numeric errors')
        # df = df.replace(-1, np.nan)
        dfs[label] = df

    # Get noise levels
    xs = list(dfs.values())[0].index.get_level_values(1).unique()

    if "implementation" in title:
        format_CONE_plot(traces['Proposed'], dfs, xs, title, subplot)
    elif ' ' in title:
        format_line_plot(dfs, xs, subplot)
    else:
        format_barplot(dfs, subplot, xs, graph_name, data_path)

    # Layout plot
    if col == 0:
        subplot.set_ylabel('Avg. accuracy (%)')
    subplot.set_xlabel('Noise (%)')
    subplot.set_title(label=f'{graph_name}', fontsize=12)


def layout_plot(fig: Figure, axes, title: str, legend_name: str):
    title_fontsize = 18

    # restrict tight_layout to the reduced area
    if "implementation" in title:  # CONE plots
        plt.tight_layout(rect=(0, 0, 0.82, 0.95))  # restrict tight_layout to the reduced area
        title_fontsize = 16
    elif ' ' in title:  # comparison of algo plots
        plt.tight_layout(rect=(0, 0, 1, 0.85))
    elif title == '':  # Single plot
        plt.tight_layout(rect=(0, 0, 1, 0.8))
    else:
        plt.tight_layout(rect=(0, 0, 1, 0.89))

    # Add legend outside the plot, top-right
    try:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        pos = [ax.get_position() for ax in axes[0]]
        center_x = (pos[0].x0 + pos[-1].x1) / 2
        if 'implementation' in title:  # CONE plots
            legend_x = pos[-1].x1 + 0.21
            legend_y = pos[-1].y1 + 0.015
        elif ' ' in title:  # Comparison plots
            legend_x = pos[-1].x1 + 0.01
            legend_y = pos[-1].y1 + 0.22
        else:  # General eval plots
            legend_x = pos[-1].x1 + 0.01
            legend_y = pos[-1].y1 + 0.18

    except TypeError:  # only one plot
        handles, labels = axes.get_legend_handles_labels()
        pos = axes.get_position()
        center_x = (pos.x0 + pos.x1) / 2
        legend_x = pos.x1 + 0.02
        legend_y = pos.y1 + 0.33

    fig.legend(handles, labels,
               loc='upper right',
               bbox_to_anchor=(legend_x, legend_y),
               bbox_transform=fig.transFigure,
               title=legend_name)

    fig.canvas.draw()  # required to update layout info

    # Only add super title for CONE plots
    if 'implementation' in title:
        plt.suptitle(title, x=center_x, fontsize=title_fontsize)


def plot_eval_graphs(grouped_traces: dict, title: str, data_file_path, legend_title: str = ''):
    traces = list(grouped_traces.values())
    number_of_plots = len(traces[0])
    assert all(len(lst) == number_of_plots for lst in traces)

    rows = 2 if len(traces[0]) != 1 else 1
    cols = math.ceil(len(traces[0]) / rows)
    if 'implementation' in title:
        figsize = (2.8 * cols, 2.3 * rows)
    else:
        figsize = (3 * cols, 3 * rows)

    # Create one figure with a grid of subplots
    fig, axes = plt.subplots(nrows=rows,
                             ncols=cols,
                             figsize=figsize,
                             sharey='row')

    for i in range(number_of_plots):
        trace_dict = {k: v[i] for k, v in grouped_traces.items()}

        row = i // cols
        col = i % cols

        subplot = axes[row, col] if number_of_plots != 1 else axes
        plot_subplot(trace_dict, subplot, col, title, data_file_path)

    layout_plot(fig, axes, title, legend_title)

    return fig


def save_fig(fig: plt.Figure, filename: str, subdir: str = ''):
    # Save plot
    root_path = get_git_root()
    path = os.path.join(root_path, 'plots', subdir, f'{filename}.pdf')
    fig.savefig(path)


original_fugal_ids = [17247, 17245, 17243, 17241, 17239, 17242]
original_grampa_ids = [22312, 22313, 22314, 22315, 22316, 22317]
original_regal_ids = [17290, 17295, 15215, 15211, 17289, 15196]
original_isorank_ids = [17269, 17268, 16096, 16098, 17270, 16095]

proposed_fugal_fixed_ids = [21209, 21476, 21485, 21490, 21574, 21612]
proposed_fugal_pca_ids = [22319, 22320, 22321, 22322, 22323, 22324]

proposed_grampa_fixed_ids = [22295, 22302, 22303, 22304, 22308, 22309]
proposed_grampa_pca_ids = [22359, 22360, 22361, 22362, 22363, 22364]

proposed_regal_fixed_ids = [22296, 22297, 22298, 22299, 22300, 22301]
proposed_regal_pca_ids = [22333, 22335, 22336, 22337, 22338, 22339]

proposed_isorank_fixed_ids = [22294, 22305, 22306, 22307, 22310, 22311]
proposed_isorank_pca_ids = [22358, 22365, 22366, 22367, 22368, 22369]


def cone_eval():
    # CONE convex initialization
    subdir = os.path.join('Other-algorithms', 'CONE')
    cone_baselines = [281, 282, 283, 284]
    trace_dict = {'Original': cone_baselines,
                  'Proposed': [14054, 14179, 14181, 14182]}
    fig = plot_eval_graphs(trace_dict, 'Convex initialization implementation', '', 'Dist scalar')
    save_fig(fig, 'CONE-convex-init-dist_scalar', subdir)

    # CONE optimal matching
    trace_dict = {'Original': cone_baselines,
                  'Proposed': [14463, 14465, 14467, 14468],
                  'lr = 0': [15161, 15163, 15164, 15165]}

    fig = plot_eval_graphs(trace_dict, 'Alignment implementation', '', 'Dist scalar')
    save_fig(fig, 'CONE-alignment-dist_scalar', subdir)


def other_algo_eval():
    # IsoRank
    subdir = os.path.join('Other-algorithms', 'IsoRank')

    # With degree similarity
    # inf-power, crime, bus, facebook 47, bio-yeast, dd
    trace_dict = {AlgorithmEnums.ORIGINAL_ISORANK: original_isorank_ids,
                  AlgorithmEnums.ISORANK_FIXED: proposed_isorank_fixed_ids,
                  AlgorithmEnums.ISORANK_PCA: proposed_isorank_pca_ids}

    fig = plot_eval_graphs(trace_dict, 'IsoRank', 'IsoRank-data.txt')
    save_fig(fig, 'IsoRank-bar-eval', subdir)

    # REGAL
    subdir = os.path.join('Other-algorithms', 'REGAL')
    # inf-power, crime, bus, facebook 47, bio-yeast, dd
    trace_dict = {AlgorithmEnums.ORIGINAL_REGAL: original_regal_ids,
                  AlgorithmEnums.REGAL_FIXED: proposed_regal_fixed_ids,
                  AlgorithmEnums.REGAL_PCA: proposed_regal_pca_ids}

    fig = plot_eval_graphs(trace_dict, 'REGAL', 'REGAL-data.txt')
    save_fig(fig, 'REGAL-bar-eval', subdir)

    # GRAMPA
    subdir = os.path.join('Other-algorithms', 'GRAMPA')
    # inf-power, crime, bus, facebook 47, bio-yeast, dd
    trace_dict = {AlgorithmEnums.ORIGINAL_GRAMPA: original_grampa_ids,
                  AlgorithmEnums.GRAMPA_FIXED: proposed_grampa_fixed_ids,
                  AlgorithmEnums.GRAMPA_PCA: proposed_grampa_pca_ids}

    fig = plot_eval_graphs(trace_dict, 'GRAMPA', 'GRAMPA-data.txt')
    save_fig(fig, 'GRAMPA-bar-eval', subdir)


def fugal_eval():
    # inf-power, crime, bus, facebook 47, bio-yeast, dd
    id_dict = {AlgorithmEnums.ORIGINAL_FUGAL: original_fugal_ids,
               AlgorithmEnums.FUGAL_FIXED: proposed_fugal_fixed_ids,
               AlgorithmEnums.FUGAL_PCA: proposed_fugal_pca_ids}

    fig = plot_eval_graphs(id_dict, 'FUGAL', 'FUGAL-primary-data-with-pca.txt')
    save_fig(fig, 'primary-eval', 'FUGAL-evaluation')

    # econ-mahindas
    id_dict = {AlgorithmEnums.ORIGINAL_FUGAL: [16385],
               AlgorithmEnums.FUGAL_FIXED: [22293],
               AlgorithmEnums.FUGAL_PCA: [22349]}
    fig = plot_eval_graphs(id_dict, '', 'FUGAL-econ-data.txt')
    save_fig(fig, 'econ-eval', 'FUGAL-evaluation')

    # email-univ, in-arenas, dublin, ca-GrQc, bio-DM-LC, arenas-meta
    id_dict = {AlgorithmEnums.ORIGINAL_FUGAL: [16388, 17253, 17244, 17251, 17238, 17254],
               AlgorithmEnums.FUGAL_FIXED: [22289, 22290, 22291, 22325, 22327, 22326],
               AlgorithmEnums.FUGAL_PCA: [22343, 22344, 22346, 22348, 22345, 22357]}

    fig = plot_eval_graphs(id_dict, 'FUGAL', 'FUGAL-appendix-data.txt')
    save_fig(fig, 'appendix-eval', 'FUGAL-evaluation')


def compare_algos():
    # FUGAL w. PCA,
    # GRAMPA w. fixed
    # IsoRank w. PCA
    # REGAL w. PCA
    id_dict = {AlgorithmEnums.FUGAL_PCA: proposed_fugal_pca_ids,
               AlgorithmEnums.GRAMPA_FIXED: proposed_grampa_fixed_ids,
               AlgorithmEnums.ISORANK_PCA: proposed_isorank_pca_ids,
               AlgorithmEnums.REGAL_PCA: proposed_regal_pca_ids}
    fig = plot_eval_graphs(id_dict, 'Proposed algorithms', '')
    save_fig(fig, 'comparison-proposed-algos', '')

    id_dict = {AlgorithmEnums.ORIGINAL_FUGAL: original_fugal_ids,
               AlgorithmEnums.ORIGINAL_GRAMPA: original_grampa_ids,
               AlgorithmEnums.ORIGINAL_ISORANK: original_isorank_ids,
               AlgorithmEnums.ORIGINAL_REGAL: original_regal_ids}
    fig = plot_eval_graphs(id_dict, 'Original algorithms', '')
    save_fig(fig, 'comparison-original-algos', '')


if __name__ == '__main__':
    fugal_eval()
    # cone_eval()
    other_algo_eval()
    compare_algos()
