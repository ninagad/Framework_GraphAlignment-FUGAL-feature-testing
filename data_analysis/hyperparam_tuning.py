# std lib imports
import os
import copy
import json
import traceback
import logging

import argparse
# lib imports
import wandb
import yaml

from data_analysis.utils import get_git_root
# local imports
from enums.scalingEnums import ScalingEnums
from enums.featureEnums import FeatureEnums
import workexp
from experiment.experiments import alggs as get_run_list, _algs
from experiment import run as run_file
from generation.generate import init1, init2


graphs = ["bio-celegans",
          "ca-netscience",
          "mammalia-voles-plj-trapping_100",
          "inf-euroroad",
          ]

noises = [0.25, 0.20, 0.15, 0.10, 0.05, 0]

iterations = 5

all_features = [FeatureEnums.DEG, FeatureEnums.CLUSTER, FeatureEnums.AVG_EGO_DEG, FeatureEnums.AVG_EGO_CLUSTER,
                   # Cluster coefficient augmented
                   FeatureEnums.SUM_EGO_CLUSTER, FeatureEnums.STD_EGO_CLUSTER, FeatureEnums.MEDIAN_EGO_CLUSTER,
                   FeatureEnums.RANGE_EGO_CLUSTER, FeatureEnums.MIN_EGO_CLUSTER, FeatureEnums.MAX_EGO_CLUSTER,
                   FeatureEnums.SKEWNESS_EGO_CLUSTER, FeatureEnums.KURTOSIS_EGO_CLUSTER,
                   # Net simile
                   FeatureEnums.EGO_EDGES, FeatureEnums.EGO_OUT_EDGES, FeatureEnums.EGO_NEIGHBORS,
                   # Miscellaneous
                   FeatureEnums.ASSORTATIVITY_EGO, FeatureEnums.INTERNAL_FRAC_EGO,
                   # degree augmented
                   FeatureEnums.SUM_EGO_DEG, FeatureEnums.STD_EGO_DEG,
                   FeatureEnums.MODE_EGO_DEGS, FeatureEnums.MEDIAN_EGO_DEGS, FeatureEnums.MIN_EGO_DEGS,
                   FeatureEnums.MAX_EGO_DEGS, FeatureEnums.RANGE_EGO_DEGS, FeatureEnums.SKEWNESS_EGO_DEGS,
                   FeatureEnums.KURTOSIS_EGO_DEGS,
                   # Centrality measures
                   FeatureEnums.CLOSENESS_CENTRALITY, FeatureEnums.DEGREE_CENTRALITY,
                   FeatureEnums.EIGENVECTOR_CENTRALITY, FeatureEnums.PAGERANK]

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


def get_hyperparam_config(run: wandb.run):
    config = run.config

    try:
        nu = config.nu
    except:
        config.nu = None
        nu = None

    mu = config.mu
    reg = config.sinkhorn_reg

    return nu, mu, reg


def log_final_metrics(graph_accs: dict, accs: list, run: wandb.run):
    nu, mu, sinkhorn_reg = get_hyperparam_config(run)

    # Log avg accuracy for each graph
    for graph in graphs:
        graph_accs = graph_accs[graph]
        graph_avg_acc = sum(graph_accs) / len(graph_accs)

        wandb.run.log({'nu': nu,
                       'mu': mu,
                       'sinkhorn_reg': sinkhorn_reg,
                       f'{graph} avg. acc': graph_avg_acc})

    # Log average accuracy across all graphs
    avg_acc = sum(accs) / len(accs)

    run.log({'nu': nu,
             'mu': mu,
             'sinkhorn_reg': sinkhorn_reg,
             'avg. accuracy': avg_acc,
             })


def setup_fugal(run: wandb.run, all_algs, features: list[FeatureEnums], log_stabilized: bool = True):
    nu, mu, sinkhorn_reg = get_hyperparam_config(run)

    if log_stabilized:
        alg_id = 22  # cuGAL with log stabilized (default)
    else:
        alg_id = 12  # FUGAL

    args_lst = [
        {'features': features,
         'nu': nu,
         'mu': mu,
         'sinkhorn_reg': sinkhorn_reg,
         'scaling': ScalingEnums.COLLECTIVE_ROBUST_NORMALIZATION,
         }
    ]

    run_lst = list(range(len(args_lst)))

    # Ensure _algs contains all algorithms before selecting algorithm
    _algs[:] = all_algs
    _algs[:] = get_run_list([alg_id, args_lst])

    return run_lst


def train(all_algs: list, feature_set: list[FeatureEnums], source_dict: dict, target_dict: dict):
    run = wandb.init(settings=wandb.Settings(start_method="thread"))
    try:
        nu, mu, sinkhorn_reg = get_hyperparam_config(run)

        # Initialize global variable artifact from run.py
        artifact_name = f'nu{nu}-mu{mu}-sinkhorn_reg{sinkhorn_reg}'
        artifact = wandb.Artifact(name=artifact_name, type='run-summary')
        run_file.artifact = artifact

        # Get arguments for fugal runs
        run_lst = setup_fugal(run, all_algs, feature_set)

        # Used to log overall avg acc and graph-wise avg. acc.
        all_accs = []
        graph_accs = {graph: [] for graph in graphs}

        for noise in noises:
            # Create an empty artifact file
            artifact_file = f'nu={nu}-mu={mu}-sinkhorn_reg={sinkhorn_reg}-noise={noise}.json'
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

                    graph_accs[graph].append(acc)

                    # Append to all accs
                    all_accs.append(acc)

            # Log current cum. accuracy for early stopping purposes
            cum_acc = sum(all_accs)
            wandb.run.log({'nu': nu,
                           'mu': mu,
                           'sinkhorn_reg': sinkhorn_reg,
                           f'cum. accuracy': cum_acc})

            # Remove artifact file from local machine
            if os.path.exists(artifact_file):
                os.remove(artifact_file)

        log_final_metrics(graph_accs, all_accs, run)

        # run summaries are logged as files in the Artifact object in run.py
        wandb.run.log_artifact(run_file.artifact).wait()

        run.finish()

    except Exception as e:
        error_trace = traceback.format_exc()  # Get full traceback as a string
        logging.error(f'{e} \n {error_trace}')
        run.finish(exit_code=1)


def initialize_sweep(sweep_config: dict, all_algs: list, sweep_name: str, feature_set: list[FeatureEnums]):

    sweep_id = wandb.sweep(sweep_config, project=sweep_name)

    source_graphs, target_graphs = generate_all_graph()

    wandb.agent(sweep_id,
                function=lambda: train(all_algs, feature_set, source_graphs, target_graphs),
                count=50
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--tune',
                        choices=['reg', 'nu_and_mu'],
                        default='nu_and_mu')

    args = parser.parse_args()
    tune = args.tune

    all_algs = copy.copy(_algs)

    # Load sweep config
    config_file = 'nu_mu_config.yaml' if tune == "nu_and_mu" else 'reg_config.yaml'
    root = get_git_root()
    path = os.path.join(root, 'data_analysis', 'tuning_configs', config_file)

    with open(path) as f:
        sweep_config = yaml.safe_load(f)

    # Select features
    # feature_set = [FeatureEnums.DEG]
    # project_name = "mu-tuning-for-degree"

    feature_set = all_features
    project_name = "reg-tuning-all-features"

    # Initialize run
    initialize_sweep(sweep_config, all_algs, project_name, feature_set)
