import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import sys

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from enums.graphEnums import GraphEnums
from enums.featureEnums import FeatureEnums, FeatureExtensions
from enums.scalingEnums import ScalingEnums
from algorithms.FUGAL.pred import feature_extraction

current_dir = (os.path.dirname(__file__))
data_dir = '../data'

graph_dict = {GraphEnums.INF_EUROROAD: 'inf-euroroad.txt',
              GraphEnums.CA_NETSCIENCE: 'ca-netscience.txt',
              GraphEnums.BIO_CELEGANS: 'bio-celegans.txt',
              GraphEnums.SOCFB_BOWDOIN47: 'socfb-Bowdoin47.txt',
              GraphEnums.VOLES: 'mammalia-voles-plj-trapping_100.txt',
              GraphEnums.MULTIMAGNA: "yeast25_Y2H1.txt",  # TODO: Check that we still use 25 version
              }
scaling = ScalingEnums.NO_SCALING

features = [FeatureEnums.DEG, FeatureEnums.CLUSTER, FeatureEnums.AVG_EGO_DEG, FeatureEnums.AVG_EGO_CLUSTER, FeatureEnums.EGO_EDGES,
            FeatureEnums.EGO_OUT_EDGES, FeatureEnums.EGO_NEIGHBORS,  # NetSimile
            FeatureEnums.SUM_EGO_CLUSTER, FeatureEnums.VAR_EGO_CLUSTER, FeatureEnums.ASSORTATIVITY_EGO, FeatureEnums.INTERNAL_FRAC_EGO,
            # Other features
            FeatureEnums.MODE_EGO_DEGS, FeatureEnums.MEDIAN_EGO_DEGS, FeatureEnums.MIN_EGO_DEGS, FeatureEnums.MAX_EGO_DEGS,
            FeatureEnums.RANGE_EGO_DEGS, FeatureEnums.SKEWNESS_EGO_DEGS, FeatureEnums.KURTOSIS_EGO_DEGS,  # Statistical features
            FeatureEnums.CLOSENESS_CENTRALITY, FeatureEnums.DEGREE_CENTRALITY, FeatureEnums.EIGENVECTOR_CENTRALITY, FeatureEnums.PAGERANK,
            # [Feature.KATZ_CENTRALITY], Centrality measures
            ]

for graph_enum, filename in graph_dict.items():
    print(graph_enum)
    path = os.path.join(current_dir, data_dir, filename)

    edges = np.loadtxt(path, int)
    graph = nx.Graph(edges.tolist())

    node_features = feature_extraction(graph, features, scaling)

    FE = FeatureExtensions()
    feature_labels = [FE.to_label(feature) for feature in features]

    feature_df = pd.DataFrame(data=node_features, columns=feature_labels)

    corr_df = feature_df.corr(method="spearman")

    # Create a mask for the upper triangle
    # mask = np.triu(np.ones_like(corr_df, dtype=bool))

    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df,
                annot=False,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                xticklabels=corr_df.columns,
                yticklabels=corr_df.columns,
                # mask=mask,
                linewidths=0.3)

    # Adjust y-axis tick labels so they don’t touch tick marks
    plt.tick_params(axis='y', pad=2, labelrotation=0)

    plt.suptitle("Spearman correlation", fontsize=24)
    graph_name = graph_enum.__repr__().lower().replace('_', '-')

    plt.title(graph_name, fontsize=12)
    plt.tight_layout()
    plt.show()
    path = os.path.join(os.path.dirname(__file__), '..', 'plots', 'correlation-test', f'{graph_name}.svg')
    plt.savefig(path)
