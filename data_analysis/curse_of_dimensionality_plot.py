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

    x_min = float('inf')
    x_max = float('-inf')
    y_max = 0  # Track the max y-value across all plots

    all_distances = []

    for feature_set in feature_sets:
        feature_matrix = feature_extraction(graph, feature_set)
        source_features, target_features = apply_scaling(feature_matrix, feature_matrix,
                                                         ScalingEnums.COLLECTIVE_ROBUST_NORMALIZATION)
        pairwise_distance_matrix = euclidean_distances(source_features, target_features)

        # Scaling of LAP term
        P = np.ones_like(pairwise_distance_matrix) / pairwise_distance_matrix.shape[0]
        lap_term = np.trace(P.T @ pairwise_distance_matrix)
        pairwise_distance_matrix *= 1 / lap_term

        all_distances.append(pairwise_distance_matrix.flatten())
        #print(f'{np.sum(pairwise_distance_matrix.flatten())=}')
        print(f'{feature_set=}')
        print(f'{np.var(pairwise_distance_matrix.flatten())=}')

    # Get global min/max
    global_min = np.min(all_distances)
    global_max = np.max(all_distances)

    # Define common bin edges (e.g., 50 bins between global_min and global_max)
    num_bins = 50
    common_bins = np.linspace(global_min, global_max, num_bins + 1)

    for i, feature_set in enumerate(feature_sets):
        # Plot histogram on subplot
        counts, bins, patches = axes[i].hist(all_distances[i], bins=common_bins, edgecolor='black')
        axes[i].set_title(f'{len(feature_set)} features')
        axes[i].set_xlabel('Distance')
        axes[i].set_ylabel('Frequency')

        x_min = min(x_min, bins[0])
        x_max = max(x_max, bins[-1])
        y_max = max(y_max, counts.max())

    # set the same y-limit for all plots
    for ax in axes:
        ax.set_ylim(0, y_max * 1.1)
        ax.set_xlim(x_min, x_max)

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
