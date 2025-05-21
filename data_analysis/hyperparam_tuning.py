# std lib imports
import os
import copy
import json
import traceback
import logging
from itertools import chain
from typing import Literal

import argparse
# lib imports
import wandb
import yaml
from wandb.errors import CommError, UsageError

from data_analysis.utils import get_git_root
# local imports
from enums.scalingEnums import ScalingEnums
from enums.featureEnums import FeatureEnums
import workexp
from experiment.experiments import alggs as get_run_list, _algs
from experiment import run as run_file
from generation.generate import init1, init2
from data_analysis.utils import get_all_features, get_forward_selected_features

graphs = ["bio-celegans",
          "ca-netscience",
          "mammalia-voles-plj-trapping_100",
          "inf-euroroad",
          ]

noises = [0.25, 0.20, 0.15, 0.10, 0.05, 0]

iterations = 5

allowed_algorithms_type = Literal['fugal', 'cugal', 'isorank', 'regal', 'grampa', 'cone']


def generate_graphs(graph_name: str):
    """
    Generates random noisy graphs separately for each noise level in 'noises'.
    Args:
        graph_name: name of graph to generate noise on
        noises: noise levels
        iterations: number of noisy graphs to generate pr. noise level

    Returns: a dictionary with noises as keys and graph lists as values.

    """
    path = f"data/{graph_name}.txt"

    source_graphs = init1([path], iterations)

    target_graph_dict = {}
    for noise in noises:
        target_graphs = init2(source_graphs, [noise])
        target_graph_dict[noise] = target_graphs

    return source_graphs, target_graph_dict


def generate_all_graph():
    # Dictionary with source graphs pr. iteration.
    source_graphs_dict = {}
    # Nested dictionary with nx graphs pr graph and noise level.
    target_graphs_dict = {}

    for graph in graphs:
        sources, targets = generate_graphs(graph)
        source_graphs_dict[graph] = sources
        target_graphs_dict[graph] = targets

    return source_graphs_dict, target_graphs_dict


def get_hyperparam_config(run: wandb.run) -> dict:
    config: wandb.sdk.wandb_config.Config = run.config
    return dict(config)


def log_final_metrics(run: wandb.run, graph_accs_dict: dict, accs: list, explained_vars: dict):
    config_dict = get_hyperparam_config(run)

    # Log avg accuracy for each graph
    for graph in graph_accs_dict.keys():
        graph_accs = graph_accs_dict[graph]
        graph_avg_acc = sum(graph_accs) / len(graph_accs)

        config_dict[f'{graph} avg. acc'] = graph_avg_acc

        # If pca is tuned, log explained variance per graph
        graph_explained_vars = explained_vars[graph]
        if graph_explained_vars:  # If not empty, compute mean
            graph_avg_var = sum(graph_explained_vars) / len(graph_explained_vars)
            config_dict[f'{graph} avg. explained var.'] = graph_avg_var

    # Log average accuracy across all graphs
    avg_acc = sum(accs) / len(accs)
    config_dict['avg. accuracy'] = avg_acc

    # For PCA, log average explained variance across graphs
    all_vars = list(chain.from_iterable(explained_vars.values()))
    if all_vars:
        avg_var = sum(all_vars) / len(all_vars)
        config_dict['avg. explained var.'] = avg_var

    run.log(config_dict)


def setup_algorithm_params(run: wandb.run, all_algs, features: list[FeatureEnums],
                           algorithm: allowed_algorithms_type):
    algo_id_dict = {'fugal': 12,
                    'cugal': 22,
                    'isorank': 6,
                    'regal': 3,
                    'grampa': 20,
                    'cone': 1}

    config_dict = get_hyperparam_config(run)

    alg_id = algo_id_dict[algorithm]

    config_dict['features'] = features
    # Add scaling to algorithms that use distance matrix.
    # No scaling for algorithms that use similarity matrix.
    # If pca is used, scaling is automatically disregarded in FUGAL
    if algorithm in ['fugal', 'cugal', 'regal', 'cone']:
        config_dict['scaling'] = ScalingEnums.COLLECTIVE_ROBUST_NORMALIZATION

    args_lst = [
        config_dict
    ]

    run_lst = list(range(len(args_lst)))

    # Ensure _algs contains all algorithms before selecting algorithm
    _algs[:] = all_algs
    _algs[:] = get_run_list([alg_id, args_lst])

    return run_lst


def train(algorithm: allowed_algorithms_type, all_algs: list, feature_set: list[FeatureEnums],
          source_dict: dict, target_dict: dict):
    run = wandb.init(settings=wandb.Settings(start_method="thread"))

    try:
        config_dict = get_hyperparam_config(run)

        # Initialize global variable artifact from run.py
        # Build artifact name with variable names and values
        artifact_name = '-'.join([f'{key}{value}' for key, value in config_dict.items()])
        artifact = wandb.Artifact(name=artifact_name, type='run-summary')
        run_file.artifact = artifact

        # Get arguments for fugal runs
        run_lst = setup_algorithm_params(run, all_algs, feature_set, algorithm)

        # Used to log overall avg acc and graph-wise avg. acc.
        all_accs = []
        graph_accs = {graph: [] for graph in graphs}
        pca_explained_vars = {graph: [] for graph in graphs}

        for noise in noises:
            # Create an empty artifact file
            artifact_file = artifact_name + f'-noise={noise}.json'

            with open(artifact_file, "w") as f:
                json.dump([], f, indent=2)

            for graph in graphs:
                nx_source_graphs = source_dict[graph]
                nx_target_graphs = target_dict[graph][noise]

                workexp.ex.run(
                    config_updates={'noises': [noise],
                                    'run': run_lst,
                                    'graph_names': [graph],
                                    'source_graphs': nx_source_graphs,
                                    'target_graphs': nx_target_graphs,
                                    'iters': iterations,
                                    'xlabel': 'Noise-level',
                                    'artifact_file': artifact_file,
                                    # 'verbose': True
                                    })

            # Add file to wandb after writing to it in run.py
            artifact.add_file(artifact_file)

            # Open file locally and append accuracies
            with open(artifact_file, 'r') as f:
                summary_dicts = json.load(f)
                for dictionary in summary_dicts:
                    graph = dictionary['graph']
                    acc = dictionary['accuracy']

                    # If we are tuning pca, log the explained variance
                    try:
                        pca_explained_var = dictionary['explained_var']
                        pca_explained_vars[graph].append(pca_explained_var)
                    except KeyError:
                        pass

                    graph_accs[graph].append(acc)

                    # Append to all accs
                    all_accs.append(acc)

            # Log current cum. accuracy for early stopping purposes
            cum_acc = sum(all_accs)

            cum_acc_dict = config_dict.copy()
            cum_acc_dict['cum. accuracy'] = cum_acc
            wandb.run.log(cum_acc_dict)

            # Remove artifact file from local machine
            if os.path.exists(artifact_file):
                os.remove(artifact_file)
        log_final_metrics(run, graph_accs, all_accs, pca_explained_vars)

        # run summaries are logged as files in the Artifact object in run.py
        wandb.run.log_artifact(run_file.artifact).wait()

        run.finish()

    except Exception as e:
        error_trace = traceback.format_exc()  # Get full traceback as a string
        logging.error(f'{e} \n {error_trace}')
        run.finish(exit_code=1)


def get_tuning_config(filename: str):
    root = get_git_root()
    path = os.path.join(root, 'data_analysis', 'tuning_configs', filename)

    with open(path) as f:
        sweep_config = yaml.safe_load(f)

    return sweep_config


def initialize_sweep(sweep_config: dict, sweep_count: int, all_algs: list, sweep_name: str,
                     feature_set: list[FeatureEnums], algorithm: allowed_algorithms_type):
    sweep_id = wandb.sweep(sweep_config, project=sweep_name)
    source_graphs, target_graphs = generate_all_graph()

    wandb.agent(sweep_id,
                function=lambda: train(algorithm, all_algs, feature_set, source_graphs, target_graphs),
                count=sweep_count
                )


def initialize_pca_sweeps(all_algos: list):
    source_graphs, target_graphs = generate_all_graph()

    # Tuning over 15 features
    features = [FeatureEnums.EGO_NEIGHBORS, FeatureEnums.MEDIAN_EGO_DEGS, FeatureEnums.EGO_OUT_EDGES,
                FeatureEnums.CLUSTER, FeatureEnums.AVG_EGO_DEG, FeatureEnums.MAX_EGO_CLUSTER,
                FeatureEnums.AVG_EGO_CLUSTER, FeatureEnums.SUM_EGO_CLUSTER, FeatureEnums.RANGE_EGO_CLUSTER,
                FeatureEnums.STD_EGO_CLUSTER, FeatureEnums.MIN_EGO_DEGS, FeatureEnums.MAX_EGO_DEGS,
                FeatureEnums.MIN_EGO_CLUSTER, FeatureEnums.MEDIAN_EGO_CLUSTER, FeatureEnums.SUM_EGO_DEG]
    sweep_dict = get_tuning_config('pca_15.yaml')

    sweep_id = wandb.sweep(sweep_dict, project='pca-15-features-tuning')
    wandb.agent(sweep_id,
                function=lambda: train('fugal', all_algos, features, source_graphs, target_graphs),
                count=15
                )

    # Tuning over 30 features
    features = get_all_features()
    sweep_dict = get_tuning_config('pca_all.yaml')

    sweep_id = wandb.sweep(sweep_dict, project='pca-all-features-tuning')
    wandb.agent(sweep_id,
                function=lambda: train('fugal', all_algos, features, source_graphs, target_graphs),
                count=30
                )


def main(tuning: str):
    all_algs = copy.copy(_algs)

    if tuning == 'pca':
        initialize_pca_sweeps(all_algs)
        return

    algorithm: allowed_algorithms_type
    config_file = f'{tuning}.yaml'

    # Load sweep config
    if (tuning == 'nu_mu_reg') or (tuning == "nu_mu"):
        algorithm = 'fugal'

        if tuning == 'nu_mu_reg':
            project_name = 'nu-mu-reg-tuning-all-features'
            feature_set = get_all_features()
            trials = 150

        if tuning == "nu_mu":
            project_name = 'nu-mu-tuning-ego-neighbors'
            feature_set = [FeatureEnums.EGO_NEIGHBORS]
            trials = 100

    elif (tuning == 'isorank') or (tuning == 'regal') or (tuning == 'grampa') or (tuning == 'cone'):
        feature_set = get_forward_selected_features()
        algorithm = tuning
        trials = 50
        project_name = ''

        if tuning == 'isorank':
            project_name = 'isorank-alpha-tuning'

        if tuning == 'regal':
            project_name = 'regal-gammaattr-tuning'

        if tuning == 'grampa':
            project_name = 'grampa-eta-tuning'

        if tuning == 'cone':
            project_name = 'cone-dist_scalar-tuning'

    else:
        raise ValueError('Unknown tuning choice')

    sweep_config = get_tuning_config(config_file)

    # Initialize run
    initialize_sweep(sweep_config, trials, all_algs, project_name, feature_set, algorithm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--tune',
                        choices=['nu_mu_reg', 'nu_mu', 'pca', 'isorank', 'regal', 'grampa', 'cone'],
                        default='nu_mu_reg')

    args = parser.parse_args()
    tune = args.tune

    main(tune)
