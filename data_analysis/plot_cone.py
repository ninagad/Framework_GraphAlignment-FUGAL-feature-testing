import os

import numpy as np
import matplotlib.pyplot as plt

from enums.graphEnums import GraphEnums
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

def plot_dist_scalar(baselines: list, sources: list):
    # Create one figure with a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2,
                             ncols=2,
                             figsize=(8, 7),
                             sharey='row')

    # Baseline trace color (orange)
    baseline_color = plt.get_cmap("Set1")(4)

    for i, (baseline, source) in enumerate(zip(baselines, sources)):
        row = 0 if i == 0 or i == 1 else 1
        col = 0 if i == 0 or i == 2 else 1

        graph_name = get_graph_names_from_file([source])[0]
        graph_name = strip_graph_name(graph_name)

        args_lst = get_algo_args(source)

        df = compute_mean_over_iters(source)

        # Get noise levels
        xs = df.index.get_level_values(1).unique()

        # Baseline
        baseline_df = compute_mean_over_iters(baseline)
        axes[row, col].plot(xs, baseline_df['mean'], label='baseline', color=baseline_color)

        # Divide df into each run
        dist_scalar_dfs = [df.iloc[i:i + 6] for i in range(0, len(df), 6)]

        colors = get_color_map(len(dist_scalar_dfs))
        marks = get_marks()

        for idx, (args, dist_scalar_df) in enumerate(zip(args_lst, dist_scalar_dfs)):
            dist_scalar = args['dist_scalar']

            axes[row, col].plot(xs, dist_scalar_df['mean'], label=dist_scalar, color=colors[idx], marker=marks[idx], markersize=5)

        # Layout plot
        axes[row, col].set_ylim(-0.1, 1.1)
        if col == 0:
            axes[row, col].set_ylabel('Accuracy')
        axes[row, col].set_xlabel('Noise level')
        axes[row, col].set_title(label=f'{graph_name}', fontsize=12)
        axes[row, col].grid(True)

    # Ensures no vertical overlap
    # fig.subplots_adjust(top=0.9, right=0.8)  # reserve space for legend on the right
    plt.tight_layout(rect=(0, 0, 0.87, 0.95))  # restrict tight_layout to the reduced area

    # Add legend outside the plot, top-left
    # Collect all legend handles and labels
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper right',
               bbox_to_anchor=(1.0, 0.908),
               title='Dist scalar')

    # Center suptitle w.r.t. plot areas excluding the tick and axis labels.
    fig.canvas.draw()  # required to update layout info
    pos = [ax.get_position() for ax in axes[0]]
    center_x = (pos[0].x0 + pos[-1].x1) / 2

    plt.suptitle(f'CONE', x=center_x, fontsize=18)

    # Save plot
    root_path = get_git_root()
    path = os.path.join(root_path, 'plots', 'Other-algorithms', 'CONE', f'CONE-dist_scalar.pdf')
    plt.savefig(path)


if __name__ == '__main__':
    source_dict = {GraphEnums.BIO_CELEGANS: 14054,
                   GraphEnums.CA_NETSCIENCE: 14179,
                   GraphEnums.INF_EUROROAD: 14181,
                   GraphEnums.VOLES: 14182}

    baseline_dict = {GraphEnums.BIO_CELEGANS: 281,
                     GraphEnums.CA_NETSCIENCE: 282,
                     GraphEnums.INF_EUROROAD: 283,
                     GraphEnums.VOLES: 284}

    plot_dist_scalar(list(baseline_dict.values()), list(source_dict.values()))
