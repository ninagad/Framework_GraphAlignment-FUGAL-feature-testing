# Standard lib imports
import os

import networkx as nx
# Lib imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# Local imports
from data_analysis.utils import get_graph, get_all_features, get_forward_selected_features, get_git_root, strip_graph_name, \
    get_fugal_features
from algorithms.FUGAL.pred import feature_extraction
from algorithms.FUGAL.Fugal import apply_scaling
from enums.featureEnums import FeatureEnums
from enums.scalingEnums import ScalingEnums


def compute_distances(graph: nx.Graph, features: list[list[FeatureEnums]]):
    all_distances = []
    if features == [get_fugal_features()]:
        scaling = ScalingEnums.NO_SCALING
        lap_scaling = False
    else:
        scaling = ScalingEnums.COLLECTIVE_ROBUST_NORMALIZATION
        lap_scaling = True

    for feature_set in features:
        feature_matrix = feature_extraction(graph, feature_set)
        source_features, target_features = apply_scaling(feature_matrix, feature_matrix,
                                                         scaling)
        pairwise_distance_matrix = euclidean_distances(source_features, target_features)

        if lap_scaling:
            # Scaling of LAP term
            P = np.ones_like(pairwise_distance_matrix) / pairwise_distance_matrix.shape[0]
            lap_term = np.trace(P.T @ pairwise_distance_matrix)
            pairwise_distance_matrix *= 1 / lap_term

        all_distances.append(pairwise_distance_matrix.flatten())

    return all_distances


def plot_distance_histograms(graph_name: str, feature_sets: list[list[FeatureEnums]]):
    graph = get_graph(f'{graph_name}.txt')

    plot_count = len(feature_sets)
    # Create one figure with 3 horizontal subplots
    fig, axes = plt.subplots(nrows=1,
                             ncols=plot_count,
                             figsize=(plot_count * 3.5, 3.5),
                             squeeze=False,
                             sharex=True,
                             sharey=True)
    axes: list[plt.Axes] = axes[0, :]

    all_distances = compute_distances(graph, feature_sets)

    # Get global min/max
    global_min = np.min(all_distances)

    percentiles_98 = [np.percentile(distances, 98) for distances in all_distances]
    largest_98_percentile = max(percentiles_98)

    # Define common bin edges (e.g., 50 bins between global_min and global_max)
    num_bins = 21

    common_bins = np.linspace(global_min, largest_98_percentile, num_bins + 1)

    for i, feature_set in enumerate(feature_sets):
        # Plot histogram on subplot
        axes[i].hist(all_distances[i], bins=common_bins, edgecolor='black')
        axes[i].set_title(f'{len(feature_set)} features')

        percentile = 98
        percentile_98 = np.percentile(all_distances[i], percentile)
        if plot_count > 1:
            # Add vertical line at the 98th percentile
            axes[i].axvline(x=percentile_98, color='grey', linestyle='--', linewidth=2, alpha=0.7,
                            label=f'{percentile}th percentile')

        axes[i].grid(True)
        axes[i].set_axisbelow(True)  # Puts grid lines behind everything

    # set the same y-limit for all plots
    #for ax in axes:
    #    #ax.set_ylim(0, y_max * 1.1)
    if plot_count > 1:
        middle_plot = 1
    else:
        middle_plot = 0

    axes[middle_plot].set_xlabel('Distance', fontsize=12)
    axes[0].set_xlim(0, largest_98_percentile)
    axes[0].set_ylabel('Frequency', fontsize=12)

    # Add legend outside the plot, top-left
    if plot_count > 1:
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1.28), ncol=3, borderaxespad=0)

    plt.tight_layout()
    fig.subplots_adjust(top=0.8)

    # Center suptitle w.r.t. plot areas excluding the tick and axis labels.
    fig.canvas.draw()  # required to update layout info
    pos = [ax.get_position() for ax in axes]
    center_x = (pos[0].x0 + pos[-1].x1) / 2

    plt.suptitle(f'{strip_graph_name(graph_name)}', x=center_x, fontsize=18)

    return fig


def save_fig(fig: plt.figure, filename: str):
    git_root = get_git_root()
    save_path = os.path.join(git_root, 'plots', 'distance-histograms', f'{filename}.pdf')
    fig.savefig(save_path)


def plot_histograms(graphs: list[str]):
    single_feature = FeatureEnums.EGO_NEIGHBORS

    for graph in graphs:
        fig = plot_distance_histograms(graph, [[single_feature],
                                               get_forward_selected_features(),
                                               get_all_features()])
        save_fig(fig, f'{strip_graph_name(graph)}-FS-features')


def plot_FUGAL_histograms(graphs: list[str]):
    fugal_features = get_fugal_features()

    for graph in graphs:
        fig = plot_distance_histograms(graph, [fugal_features])
        save_fig(fig, f'{strip_graph_name(graph)}-FUGAL-features')


if __name__ == "__main__":
    graph_names = ['bio-celegans',
                   'inf-euroroad',
                   'ca-netscience',
                   'mammalia-voles-plj-trapping_100'
                   ]

    #plot_FUGAL_histograms(graph_names)
    plot_histograms(graph_names)
