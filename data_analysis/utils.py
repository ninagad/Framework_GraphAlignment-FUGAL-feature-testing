import subprocess
from pathlib import Path
import os
import json
from typing import Literal

import numpy as np
import networkx as nx
import pandas as pd

from enums.featureEnums import FeatureEnums


def get_git_root():
    try:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
        return Path(root)
    except subprocess.CalledProcessError:
        raise RuntimeError("Not a git repository")


def get_eval_graphs():
    graphs = ["inf-power", "ia-crime-moreno", "power-685-bus", "socfb-Bowdoin47", "bio-yeast", "DD_g501"]

    return graphs


def get_appendix_eval_graphs():
    graphs = ["email-univ", "in-arenas", "infect-dublin", "tomography", "econ-mahindas"]

    return graphs


def get_eval_graph_run_ids():
    ids = {"inf-power": 17236,
           "ia-crime-moreno": 16121,
           "power-685-bus": 15142,
           "socfb-Bowdoin47": 15126,
           "bio-yeast": 17235,
           "DD_g501": 15129
           }

    return ids


def get_appendix_eval_graph_run_ids():
    ids = {"email-univ": 16377,
           "in-arenas": 17253,
           "infect-dublin": 15123,
           "tomography": 17264,
           "econ-mahindas": 16374
           }

    return ids


def get_graph(filename: str) -> nx.Graph:
    root = get_git_root()
    data_dir = os.path.join(root, 'data')

    graph_path = os.path.join(data_dir, filename)

    edges = np.loadtxt(graph_path, int)
    graph = nx.Graph(edges.tolist())

    return graph


def get_all_features() -> [FeatureEnums]:
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


def get_fugal_features() -> [FeatureEnums]:
    fugal_features = [FeatureEnums.DEG,
                      FeatureEnums.CLUSTER,
                      FeatureEnums.AVG_EGO_DEG,
                      FeatureEnums.AVG_EGO_CLUSTER]

    return fugal_features


def get_forward_selected_features() -> [FeatureEnums]:
    feature_set = [FeatureEnums.EGO_NEIGHBORS,
                   FeatureEnums.SUM_EGO_DEG,
                   FeatureEnums.STD_EGO_DEG,
                   FeatureEnums.DEGREE_CENTRALITY]

    return feature_set

def get_15_features() -> [FeatureEnums]:
    features_set = [FeatureEnums.EGO_NEIGHBORS, FeatureEnums.MEDIAN_EGO_DEGS, FeatureEnums.EGO_OUT_EDGES,
                FeatureEnums.CLUSTER, FeatureEnums.AVG_EGO_DEG, FeatureEnums.MAX_EGO_CLUSTER,
                FeatureEnums.AVG_EGO_CLUSTER, FeatureEnums.SUM_EGO_CLUSTER, FeatureEnums.RANGE_EGO_CLUSTER,
                FeatureEnums.STD_EGO_CLUSTER, FeatureEnums.MIN_EGO_DEGS, FeatureEnums.MAX_EGO_DEGS,
                FeatureEnums.MIN_EGO_CLUSTER, FeatureEnums.MEDIAN_EGO_CLUSTER, FeatureEnums.SUM_EGO_DEG]

    return features_set

def get_training_graph_names():
    graphs = ['bio-celegans', 'ca-netscience', 'inf-euroroad', 'voles']
    return graphs


def strip_graph_name(name: str) -> str:
    if 'voles' in name:
        return 'voles'
    else:
        return name


def get_acc_file_as_df(run: int) -> pd.DataFrame:
    root = get_git_root()
    path = os.path.join(root, 'Server-runs', f'{run}', 'res', 'acc.xlsx')

    df = pd.read_excel(path, index_col=[0, 1])
    df.index.names = ['Feature', 'Noise']

    return df


def get_acc_files_as_single_df(runs: list[int]) -> pd.DataFrame:
    dfs = []
    for run in runs:
        df = get_acc_file_as_df(run)
        dfs.append(df)

    # Stack dfs
    df = pd.concat(dfs, axis=0)
    return df


def get_config_file(run: int) -> json:
    root = get_git_root()
    path = os.path.join(root, 'Server-runs', f'{run}', 'config.json')

    config = json.load(open(path))

    return config


def get_graph_names_from_file(runs: list[int]) -> list[str]:
    graph_names = []
    for run in runs:
        config = get_config_file(run)

        names = config['graph_names']
        if len(names) != 1:
            raise NotImplementedError

        graph_names.append(names[0])

    return graph_names

def get_algorithm(run: int) -> str:
    config = get_config_file(run)
    algorithm = set(config['algs'][0][0].values())
    if len(algorithm) != 1:
        raise NotImplementedError

    name = str(algorithm).split('.')[1]
    return name

def get_algo_args(run: int) -> list[dict]:
    config = get_config_file(run)
    args = [execution[1] for execution in config['algs']]

    return args


def get_parameter(run: int, param: Literal['nu', 'mu', 'sinkhorn_reg']) -> list[float] | float:
    args = get_algo_args(run)

    param_vals = [args_dict[param] for args_dict in args]

    unique_param_vals = set(param_vals)

    if len(unique_param_vals) != 1:
        return param_vals
    else:
        return param_vals[0]
