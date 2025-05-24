from typing import Literal

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import sys

from enums.graphEnums import GraphEnums
from enums.featureEnums import FeatureEnums, FeatureExtensions
from enums.scalingEnums import ScalingEnums
from algorithms.FUGAL.pred import feature_extraction
from algorithms.FUGAL.Fugal import apply_scaling
from data_analysis.utils import get_all_features, get_graph, get_git_root

graph_dict = {GraphEnums.INF_EUROROAD: 'inf-euroroad.txt',
              GraphEnums.CA_NETSCIENCE: 'ca-netscience.txt',
              GraphEnums.BIO_CELEGANS: 'bio-celegans.txt',
              GraphEnums.VOLES: 'mammalia-voles-plj-trapping_100.txt',
              GraphEnums.SOCFB_BOWDOIN47: 'socfb-Bowdoin47.txt',
              GraphEnums.MULTIMAGNA: "yeast25_Y2H1.txt",
              }

scaling = ScalingEnums.COLLECTIVE_ROBUST_NORMALIZATION

features = get_all_features()


def compute_features(graphs: dict):
    feature_dfs = []

    for graph_enum, filename in graphs.items():
        print(graph_enum)

        graph = get_graph(filename)

        node_features = feature_extraction(graph, features)
        node_features, _ = apply_scaling(node_features, node_features, scaling)

        FE = FeatureExtensions()
        feature_labels = [FE.to_label(feature) for feature in features]

        feature_df = pd.DataFrame(data=node_features, columns=feature_labels)
        feature_dfs.append(feature_df)

    return feature_dfs


def compute_correlation(features: pd.DataFrame, method: Literal["pearson", "kendall", "spearman"]):
    corr_df = features.corr(method=method)
    corr_df.index = corr_df.columns

    return corr_df


def plot_correlations(save_path: str, correlations: pd.DataFrame, title: str, subtitle: str):
    # Create a mask for the upper triangle
    # mask = np.triu(np.ones_like(corr_df, dtype=bool))

    # Create heatmap
    plt.figure(figsize=(8, 7))
    sns.heatmap(correlations,
                annot=False,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                xticklabels=correlations.columns,
                yticklabels=correlations.index,
                # mask=mask,
                linewidths=0.3)

    # Adjust y-axis tick labels so they don’t touch tick marks
    plt.tick_params(axis='y', pad=2, labelrotation=0)

    plt.suptitle(title, fontsize=24)

    plt.title(subtitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)


def plot_correlation_over_forward_selection_graphs():
    method: Literal["pearson", "kendall", "spearman"] = 'spearman'
    title = f'{method} correlation'.capitalize()
    subtitle = ''

    git_root = get_git_root()
    save_path = os.path.join(git_root, 'plots', 'correlation-test', f'{method}-correlation_forward_selection.svg')

    forward_selection_graphs = [GraphEnums.BIO_CELEGANS,
                                GraphEnums.CA_NETSCIENCE,
                                GraphEnums.INF_EUROROAD,
                                GraphEnums.VOLES]

    graphs = {graph: graph_dict[graph] for graph in forward_selection_graphs}

    feature_dfs = compute_features(graphs)
    feature_df = pd.concat(feature_dfs)
    correlations_df = compute_correlation(feature_df, method)

    fe = FeatureExtensions()
    ego_neighbors_label = fe.to_label(FeatureEnums.EGO_NEIGHBORS)
    ego_neighbors_df = correlations_df[ego_neighbors_label].to_frame()

    plot_correlations(save_path, ego_neighbors_df, title, subtitle)


def plot_correlations_for_individual_graphs():
    title = 'Spearman correlation'
    git_root = get_git_root()

    for graph_enum, file in graph_dict.items():
        graph_name = graph_enum.__repr__().lower().replace('_', '-')
        subtitle = graph_name

        save_path = os.path.join(git_root, 'plots', 'correlation-test', f'{graph_name}.svg')

        feature_df = compute_features({graph_enum: file})[0]
        correlation_df = compute_correlation(feature_df, 'spearman')
        plot_correlations(save_path, correlation_df, title, subtitle)


if __name__ == '__main__':
    plot_correlation_over_forward_selection_graphs()
    #plot_correlations_for_individual_graphs()
