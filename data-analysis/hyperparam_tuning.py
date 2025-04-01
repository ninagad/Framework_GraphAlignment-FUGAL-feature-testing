import copy
import json
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
    graph_accs = {}
    try:
        config = run.config
        mu = config.mu

        artifact_name = f'{mu}'

        # Initialize global variable artifact from run.py
        run_file.artifact = wandb.Artifact(name=artifact_name, type='run-summary')

        graphs = [("bio-celegans", 36),
                  ("ca-netscience", 19),
                  ("mammalia-voles-plj-trapping_100", 60),
                  ("inf-euroroad", 20),
                  ]

        noises = [0, 0.05, 0.10, 0.15, 0.20, 0.25]

        iterations = 5

        alg_id = 12  # FUGAL
        args_lst = [
            {'features': feature_set,
             'mu': mu,
             'scaling': ScalingEnums.COLLECTIVE_ROBUST_NORMALIZATION,
             }
        ]

        run_lst = list(range(len(args_lst)))

        # Ensure _algs contains all algorithms before selecting algorithm
        _algs[:] = all_algs
        _algs[:] = get_run_list([alg_id, args_lst])

        for graph, baseline in graphs:
            graph_accs[graph] = []  # Used for logging accuracy later
            graph_wrapper = get_graph_paths([graph])

            workexp.ex.run(
                config_updates={'noises': noises,
                                'run': run_lst,
                                'graph_names': [graph],
                                'load': [baseline, baseline],
                                'graphs': graph_wrapper,
                                'iters': iterations,
                                'xlabel': 'Noise-level',
                                #'verbose': True
                                })

        # run summaries are logged as files in the Artifact object in run.py
        wandb.run.log_artifact(run_file.artifact).wait()

        # Access the logged artifact
        artifact = wandb.run.use_artifact(f'{artifact_name}:latest')

        # Download the artifact to a local folder
        artifact_dir = artifact.download()

        # Compute avg accuracy across all runs and log
        accs = []
        artifact_files = os.listdir(artifact_dir)
        for file in artifact_files:
            with open(os.path.join(artifact_dir, file), 'r') as f:
                summary_dict = json.load(f)
                graph = summary_dict['graph']
                acc = summary_dict['accuracy']
                # Append to acc list for specific graph
                graph_accs[graph].append(acc)

                # Append to all accs
                accs.append(acc)

        avg_acc = sum(accs) / len(accs)

        run.log({'mu': mu,
                 'Avg. accuracy': avg_acc})

        for graph, acc_lst in graph_accs.items():
            graph_avg_acc = sum(acc_lst) / len(acc_lst)
            run.log({'mu': mu,
                     f'{graph} avg. acc': graph_avg_acc})

        run.finish()

    except Exception as e:
        error_trace = traceback.format_exc()  # Get full traceback as a string
        logging.error(f'{e} \n {error_trace}')
        run.finish(exit_code=1)



def initialize_sweep(all_algs: list, sweep_name: str, feature_set: list[FeatureEnums]):
    sweep_config = {
        "method": "bayes",  # Bayesian optimization for mu
        "metric": {"name": "Avg. accuracy", "goal": "maximize"},
        "parameters": {
            "mu": {"min": 0.01,
                   "max": 200.0,
                   # Rounded log uniform distribution (more values for small mu)
                   'distribution': 'q_log_uniform_values',
                   "q": 0.01  # Restrict to 2 decimal precision for mu
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

    feature_set = [FeatureEnums.DEG]
    project_name = "mu-tuning-for-degree"

    initialize_sweep(all_algs, project_name, feature_set)
