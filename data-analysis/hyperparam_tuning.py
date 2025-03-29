import copy
import json
import wandb
import sys
import os

# Get the root directory and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import workexp
from experiment.experiments import alggs as get_run_list, _algs, rgraphs as get_graph_paths
from enums.scalingEnums import ScalingEnums
from enums.featureEnums import FeatureEnums
from experiment import run as run_file

def train(all_algs: list):
    run = wandb.init(settings=wandb.Settings(start_method="thread"))
    config = run.config
    mu = config.mu

    artifact_name = f'{mu}'

    # Initialize global variable artifact from run.py
    run_file.artifact = wandb.Artifact(name=artifact_name, type='run-summary')

    # TODO: Which graphs should we include?
    graph_names = ["bio-celegans",
                   "ca-netscience",
                   "mammalia-voles-plj-trapping_100",
                   "yeast25_Y2H1",
                   "inf-euroroad",
                   "socfb-Bowdoin47",
                   ]

    noises = [0, 0.05, 0.10, 0.15, 0.20, 0.25]

    feature_set = [FeatureEnums.DEG, FeatureEnums.CLUSTER]  # TODO: choose feature set

    iterations = 5

    alg_id = 12  # FUGAL
    args_lst = [
        {'features': feature_set,
         'mu': mu,
         'scaling': ScalingEnums.NORMALIZE_FEATURES,  # TODO: DECIDE SCALING
         }  # for feature in features
    ]

    run_lst = list(range(len(args_lst)))

    # Ensure _algs contains all algorithms before selecting algorithm
    _algs[:] = all_algs
    _algs[:] = get_run_list([alg_id, args_lst])

    graph_wrapper = get_graph_paths(graph_names)

    # TODO: make sure to load the same graphs for each mu.
    #  Generate them by running a single time and load from there
    workexp.ex.run(  # named_configs=['hyperparam_setup'],
        config_updates={'noises': noises,
                        'run': run_lst,
                        'graph_names': graph_names,
                        'graphs': graph_wrapper,
                        'iters': iterations,
                        'xlabel': 'Noise-level',
                        'verbose': True
                        })

    # run summaries are logged as files in the Artifact object in run.py
    wandb.run.log_artifact(run_file.artifact).wait()

    # Access the logged artifact
    artifact = wandb.run.use_artifact(f'{artifact_name}:latest')

    # Download the artifact to a local folder
    artifact_dir = artifact.download()

    # Read the file from the artifact
    artifact_dir = f"{artifact_dir}"

    # Compute avg accuracy across all runs and log
    acc_sum = 0
    artifact_files = os.listdir(artifact_dir)
    for file in artifact_files:

        with open(os.path.join(artifact_dir, file), 'r') as f:
            summary_dict = json.load(f)

            acc_sum += summary_dict['accuracy']

    acc_avg = acc_sum / len(artifact_files)

    run.log({'mu': mu,
             'Avg. accuracy': acc_avg})

    run.finish()


if __name__ == "__main__":
    all_algs = copy.copy(_algs)

    sweep_config = {
        "method": "bayes",  # Bayesian optimization for mu
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "mu": {"min": 0.01,
                   "max": 10000.0,
                   'distribution': 'q_uniform', # Rounded uniform distribution
                   "q": 0.01  # Restrict to 2 decimal precision for mu
                   },
        }
        # TODO: try hyperband (bayesian bandits) optimization instead should run faster
        #  -> we can't since we only log one value per hyperparameter setting,
        #  so early stopping is not possible?

        # TODO: try log/exp scaled distribution for mu
    }
    # TODO: ENSURE THAT IT TERMINATES IMMEDIATELY IF A NUMERIC ERROR OCCURS
    #  -> this is extremely harmful and should never happen in a setting where
    #  you actually want to use the algorithm to align graphs.

    sweep_id = wandb.sweep(sweep_config, project="mu-hyperparameter-tuning")

    wandb.agent(sweep_id,
                function=lambda: train(all_algs=all_algs)
                )
