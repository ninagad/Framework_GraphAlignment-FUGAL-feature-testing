import subprocess
from pathlib import Path
import os

import numpy as np
import networkx as nx

from enums.featureEnums import FeatureEnums


def get_git_root():
    try:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
        return Path(root)
    except subprocess.CalledProcessError:
        raise RuntimeError("Not a git repository")


def get_graph(filename: str):
    current_dir = (os.path.dirname(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')

    graph_path = os.path.join(data_dir, filename)

    edges = np.loadtxt(graph_path, int)
    graph = nx.Graph(edges.tolist())

    return graph


def get_all_features():
    all_features = [FeatureEnums.DEG, FeatureEnums.CLUSTER, FeatureEnums.AVG_EGO_DEG, FeatureEnums.AVG_EGO_CLUSTER,
                    # Net simile
                    FeatureEnums.EGO_EDGES, FeatureEnums.EGO_OUT_EDGES, FeatureEnums.EGO_NEIGHBORS,
                    # degree augmented
                    FeatureEnums.SUM_EGO_DEG, FeatureEnums.STD_EGO_DEG,
                    FeatureEnums.MODE_EGO_DEGS, FeatureEnums.MEDIAN_EGO_DEGS, FeatureEnums.MIN_EGO_DEGS,
                    FeatureEnums.MAX_EGO_DEGS, FeatureEnums.RANGE_EGO_DEGS, FeatureEnums.SKEWNESS_EGO_DEGS,
                    FeatureEnums.KURTOSIS_EGO_DEGS,
                    # Cluster coefficient augmented
                    FeatureEnums.SUM_EGO_CLUSTER, FeatureEnums.STD_EGO_CLUSTER, FeatureEnums.MEDIAN_EGO_CLUSTER,
                    FeatureEnums.RANGE_EGO_CLUSTER, FeatureEnums.MIN_EGO_CLUSTER, FeatureEnums.MAX_EGO_CLUSTER,
                    FeatureEnums.SKEWNESS_EGO_CLUSTER, FeatureEnums.KURTOSIS_EGO_CLUSTER,
                    # Miscellaneous
                    FeatureEnums.ASSORTATIVITY_EGO, FeatureEnums.INTERNAL_FRAC_EGO,
                    # Centrality measures
                    FeatureEnums.CLOSENESS_CENTRALITY, FeatureEnums.DEGREE_CENTRALITY,
                    FeatureEnums.EIGENVECTOR_CENTRALITY, FeatureEnums.PAGERANK]

    return all_features


def get_fugal_features():
    fugal_features = [FeatureEnums.DEG,
                      FeatureEnums.CLUSTER,
                      FeatureEnums.AVG_EGO_DEG,
                      FeatureEnums.AVG_EGO_CLUSTER]

    return fugal_features
