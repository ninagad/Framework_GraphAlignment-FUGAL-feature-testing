import copy
import json

import numpy as np
import wandb
import traceback
import logging
import sys
import os

# Get the root directory and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import workexp
from experiment.experiments import alggs as get_run_list, _algs, rgraphs as get_graph_paths
from enums.scalingEnums import ScalingEnums
from enums.featureEnums import FeatureEnums
from experiment import run as run_file


def train(all_algs: list, feature_set: list[FeatureEnums]):
    run = wandb.init(settings=wandb.Settings(start_method="thread"))
    try:
        config = run.config
        nu = config.nu
        mu = config.mu
        sinkhorn_reg = config.sinkhorn_reg

        artifact_name = f'nu{nu}-mu{mu}-sinkhorn_reg{sinkhorn_reg}'

        # Initialize global variable artifact from run.py
        artifact = wandb.Artifact(name=artifact_name, type='run-summary')
        run_file.artifact = artifact

        # Create an empty artifact file
        artifact_file = f'nu={nu}-mu={mu}-sinkhorn_reg={sinkhorn_reg}.json'
        with open(artifact_file, "w") as f:
            json.dump([], f, indent=2)

        graphs = [("bio-celegans", 689),
                  ("ca-netscience", 690),
                  ("mammalia-voles-plj-trapping_100", 692),
                  ("inf-euroroad", 691),
                  ]

        noises = [0, 0.05, 0.10, 0.15, 0.20, 0.25]

        iterations = 5

        alg_id = 12  # FUGAL
        args_lst = [
            {'features': feature_set,
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

        for graph, baseline in graphs:
            graph_wrapper = get_graph_paths([graph])

            workexp.ex.run(
                config_updates={'noises': noises,
                                'run': run_lst,
                                'graph_names': [graph],
                                'load': [baseline, baseline],
                                'graphs': graph_wrapper,
                                'iters': iterations,
                                'xlabel': 'Noise-level',
                                # 'verbose': True
                                })

        # Add file to wandb after writing to it in run.py
        artifact.add_file(artifact_file)

        # run summaries are logged as files in the Artifact object in run.py
        wandb.run.log_artifact(run_file.artifact).wait()

        # Access the logged artifact
        artifact = wandb.run.use_artifact(f'{artifact_name}:latest')

        # Download the artifact to a local folder
        artifact_dir = artifact.download()

        # Compute avg accuracy across all runs and log
        accs = []
        with open(os.path.join(artifact_dir, artifact_file), 'r') as f:
            summary_dicts = json.load(f)
            for dictionary in summary_dicts:
                acc = dictionary['accuracy']

                # Append to all accs
                accs.append(acc)

        avg_acc = sum(accs) / len(accs)

        run.log({'nu': nu,
                 'mu': mu,
                 'sinkhorn_reg': sinkhorn_reg,
                 'Avg. accuracy': avg_acc})

        run.finish()

        # Remove artifact file from local machine
        if os.path.exists(artifact_file):
            os.remove(artifact_file)

    except Exception as e:
        error_trace = traceback.format_exc()  # Get full traceback as a string
        logging.error(f'{e} \n {error_trace}')
        run.finish(exit_code=1)


def initialize_sweep(all_algs: list, sweep_name: str, feature_set: list[FeatureEnums]):
    # Generate 100 values evenly spaced between 0 and 1 and 100 values evenly spaced between 1 and 100.
    zero_to_one_values = 0.01 * np.arange(1, 100)
    one_to_100_values = np.arange(1, 101)
    hyper_param_values = np.hstack((zero_to_one_values, one_to_100_values)).tolist()

    sweep_config = {
        "method": "bayes",  # Bayesian optimization for mu
        "metric": {"name": "Avg. accuracy", "goal": "maximize"},
        "parameters": {
            "nu": {'values': hyper_param_values},
                # {"min": 0.01,
                #    "max": 100,
                #    'distribution': 'q_log_uniform_values',  # Rounded log distribution to try more small values.
                #    "q": 0.01  # Restrict to 2 decimal precision
                #    },
            "mu": {'values': hyper_param_values},
                # {"min": 0.01,
                #    "max": 100.0,
                #    'distribution': 'q_log_uniform_values',  # Rounded log distribution to try more small values.
                #    "q": 0.01  # Restrict to 2 decimal precision
                #    },
            "sinkhorn_reg": {"min": 0.001,
                             "max": 1,
                             "distribution": 'q_uniform',
                             "q": 0.001
                             },
        }
        # TODO: try hyperband (bayesian bandits) optimization instead should run faster
        #  -> we can't since we only log one value per hyperparameter setting,
        #  so early stopping is not possible?
    }

    sweep_id = wandb.sweep(sweep_config, project=sweep_name)

    wandb.agent(sweep_id,
                function=lambda: train(all_algs, feature_set),
                count=50
                )


if __name__ == "__main__":
    all_algs = copy.copy(_algs)

    # feature_set = [FeatureEnums.DEG]
    # project_name = "mu-tuning-for-degree"

    feature_set = [FeatureEnums.DEG, FeatureEnums.CLUSTER, FeatureEnums.AVG_EGO_DEG, FeatureEnums.AVG_EGO_CLUSTER,
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
    project_name = "nu-mu-reg-tuning-all-features"

    initialize_sweep(all_algs, project_name, feature_set)
