# Standard lib imports
import os

# Lib imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# Local imports
from utils import get_graph, get_all_features, get_fugal_features
from algorithms.FUGAL.pred import feature_extraction
from algorithms.FUGAL.Fugal import apply_scaling
from enums.featureEnums import FeatureEnums
from enums.scalingEnums import ScalingEnums


def plot_distance_histograms(graph_name: str):
    graph = get_graph(f'{graph_name}.txt')

    single_feature = FeatureEnums.EGO_NEIGHBORS

    feature_sets = [[single_feature],
                    get_fugal_features(),
                    get_all_features()]

    # Create one figure with 3 horizontal subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    y_max = 0  # Track the max y-value across all plots

    for i, feature_set in enumerate(feature_sets):
        feature_matrix = feature_extraction(graph, feature_set)
        source_features, target_features = apply_scaling(feature_matrix, feature_matrix, ScalingEnums.COLLECTIVE_ROBUST_NORMALIZATION)

        # axes[0].hist(feature_matrix.flatten())
        # axes[1].hist(source_features.flatten())
        # plt.show()
        pairwise_distance_matrix = euclidean_distances(source_features, target_features)

        # Plot histogram on subplot
        counts, bins, patches = axes[i].hist(pairwise_distance_matrix.flatten(), bins=50, edgecolor='black')
        axes[i].set_title(f'{len(feature_set)} features')
        axes[i].set_xlabel('Distance')
        axes[i].set_ylabel('Frequency')

        # Update y_max if this plot has higher bars
        y_max = max(y_max, counts.max())

    # set the same y-limit for all plots
    for ax in axes:
        ax.set_ylim(0, y_max * 1.1)

    plt.tight_layout()

    current_dir = os.path.dirname(__file__)
    save_path = os.path.join(current_dir, '..', 'plots', 'distance-histograms', f'{graph_name}-histograms.svg')
    fig.savefig(save_path)


if __name__ == "__main__":
    graph_names = ['bio-celegans',
                   'inf-euroroad',
                   'ca-netscience',
                   'mammalia-voles-plj-trapping_100']

    for graph in graph_names:
        plot_distance_histograms(graph)