import subprocess
from enum import Enum
from pathlib import Path
import os
import json
from typing import Literal

import numpy as np
import networkx as nx
import pandas as pd

from enums.featureEnums import FeatureEnums


class AlgorithmEnums(Enum):
    ORIGINAL_FUGAL = 'FUGAL (original)'
    FUGAL_FIXED = 'FUGAL w. fixed'
    FUGAL_PCA = 'FUGAL w. PCA'

    ORIGINAL_REGAL = 'REGAL (original)'
    REGAL_FIXED = 'REGAL w. fixed'
    REGAL_PCA = 'REGAL w. PCA'

    ORIGINAL_GRAMPA = 'GRAMPA (original)'
    GRAMPA_FIXED = 'GRAMPA w. fixed'
    GRAMPA_PCA = 'GRAMPA w. PCA'

    ORIGINAL_ISORANK = 'IsoRank (original)'
    ISORANK_FIXED = 'IsoRank w. fixed'
    ISORANK_PCA = 'IsoRank w. PCA'


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


def get_metric_noise_graph_run_ids():
    ids = {"inf-power": 17236,
           "ia-crime-moreno": 16121
           }

    return ids


def get_appendix_eval_graph_run_ids():
    ids = {"email-univ": 16377,
           "in-arenas": 17253,
           "infect-dublin": 15123,
           "bio-DM-LC_no_weight": 17233,
           "ca-CrQc": 17251,
           "arenas-meta": 17253,
           # "tomography": 17264,
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
    if 'no_weight' in name:
        name = name.split('_')[0]
        return name
    else:
        return name


def get_acc_file_as_df(run: int, dir: Literal['Skadi-runs', 'Server-runs'] = 'Server-runs') -> pd.DataFrame:
    root = get_git_root()
    path = os.path.join(root, dir, f'{run}', 'res', 'acc.xlsx')

    df = pd.read_excel(path, index_col=[0, 1])
    df.index.names = ['Feature', 'Noise']

    return df


def get_total_time_as_df(run: int) -> pd.DataFrame:
    root = get_git_root()
    path = os.path.join(root, 'Time-runs', f'{run}', 'res')
    alg_time_path = os.path.join(path, 'time_alg.xlsx')
    matching_time_path = os.path.join(path, 'time_matching.xlsx')

    alg_time_df = pd.read_excel(alg_time_path, index_col=[0, 1])
    alg_time_df.index.names = ['Feature', 'Noise']

    matching_time_df = pd.read_excel(matching_time_path, index_col=[0, 1])
    matching_time_df.index.names = ['Feature', 'Noise']

    total_time_df = alg_time_df + matching_time_df

    return total_time_df


def get_acc_files_as_single_df(runs: list[int]) -> pd.DataFrame:
    dfs = []
    for run in runs:
        df = get_acc_file_as_df(run)
        dfs.append(df)

    # Stack dfs
    df = pd.concat(dfs, axis=0)
    return df


def get_config_file(run: int, dir: Literal['Server-runs', 'Time-runs', 'Skadi-runs'] = 'Server-runs') -> json:
    root = get_git_root()
    path = os.path.join(root, dir, f'{run}', 'config.json')

    config = json.load(open(path))

    return config


def get_metric(run: int, dir: Literal['Server-runs', 'Skadi-runs'] = 'Server-runs') -> str:
    config = get_config_file(run, dir)
    metric = config['accs']

    if len(metric) != 1:
        raise ValueError(f'Expected a single metric, but got {metric}')

    metric_names = {0: 'Accuracy',
                    3: '$S^3$',
                    5: 'MNC',
                    6: 'Frobenius norm'}

    return metric_names[metric[0]]


def get_noise_type(run: int, dir: Literal['Server-runs', 'Skadi-runs']='Server-runs'):
    config = get_config_file(run, dir)
    try:
        noise_type = config['noise_type']
    except KeyError:
        # If the noise-type is not specified, it is one-way (default).
        return 'One-way'

    if len(noise_type) != 1:
        raise ValueError(f'Expected a single noise type, but got {noise_type}')

    noise_type_names = {1: 'One-way',
                        2: 'Multi-modal',
                        3: 'Two-way'}

    return noise_type_names[noise_type[0]]


def get_graph_names_from_file(runs: list[int], dir: Literal['Server-runs', 'Time-runs', 'Skadi-runs'] = 'Server-runs') -> list[str]:
    graph_names = []
    for run in runs:
        config = get_config_file(run, dir)

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


def eval_bar_plot_palette():
    # Greens
    baseline_color = '#b1de89'
    fixed_color = '#31a354'
    pca_color = '#3c5a3a'
    palette = [baseline_color, fixed_color, pca_color]

    return palette
