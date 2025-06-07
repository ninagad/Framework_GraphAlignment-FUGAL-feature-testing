from typing import Literal

import workexp
from data_analysis.utils import get_forward_selected_features, get_15_features, get_fugal_features
from enums.scalingEnums import ScalingEnums
from experiment.experiments import alggs as get_run_list, _algs, get_graph_paths as get_graph_paths

allowed_algorithms = Literal['fugal', 'cugal', 'isorank', 'regal', 'grampa', 'cone']

eval_graphs = Literal['inf-power', 'ia-crime-moreno', 'power-685-bus', 'socfb-Bowdoin47', 'bio-yeast', 'DD_g501']


def get_algo_id(algorithm: allowed_algorithms):
    algo_id_dict = {'fugal': 12,
                    'cugal': 22,
                    'isorank': 6,
                    'regal': 3,
                    'grampa': 20,
                    'cone': 1}

    return algo_id_dict[algorithm]


def get_proposed_fugal_w_pca_arguments():
    args_lst = [
        {'features': get_15_features(),
         'nu': 447.24,
         'mu': 442.66,
         'sinkhorn_reg': 0.00141,
         'scaling': ScalingEnums.NO_SCALING,
         'pca_components': 8,
         'frank_wolfe_iters': 2,
         }
    ]
    return args_lst


def get_proposed_fugal_w_fixed_feature_set_arguments():
    args_lst = [
        {'features': get_forward_selected_features(),
         'nu': 447.24,
         'mu': 442.66,
         'sinkhorn_reg': 0.00141,
         'scaling': ScalingEnums.COLLECTIVE_ROBUST_NORMALIZATION,
         'frank_wolfe_iters': 2,
         }
    ]
    return args_lst

def get_original_fugal_args(mu: float):
    args_lst = [
        {'features': get_fugal_features(),
         'mu': mu,
         }
    ]
    return args_lst

def get_original_fugal_eval_graph_mus():
    mus = {"inf-power": 0.8981,
           "ia-crime-moreno": 0.8797,
           "power-685-bus": 0.8597,
           "socfb-Bowdoin47": 0.7382,
           "bio-yeast": 0.8917,
           "DD_g501": 0.8387
           }

    return mus

def save_config_info(path, info):
    # Write to file
    with open(path, "a") as file:
        file.write(f'\n{info}')


def run_alg(save_file: str,
            algorithm: allowed_algorithms,
            all_algs: list,
            algo_args: list[dict],
            graph: str,
            load_graph: int | None = None,
            noise_type: int = 1,
            acc: int = 0):
    noises = [0, 0.05, 0.10, 0.15, 0.20, 0.25]

    iterations = 5

    run_lst = list(range(len(algo_args)))

    # Ensure _algs contains all algorithms before selecting algorithm
    _algs[:] = all_algs.copy()

    _algs[:] = get_run_list([get_algo_id(algorithm), algo_args])

    graph_wrapper = get_graph_paths([graph])

    if load_graph is None:
        load = []
    else:
        load = [load_graph, load_graph]

    run = workexp.ex.run(
        config_updates={'noises': noises,
                        'run': run_lst,
                        'graph_names': [graph],
                        'load': load,
                        'graphs': graph_wrapper,
                        'iters': iterations,
                        'xlabel': 'Noise-level',
                        'verbose': True,
                        'noise_type': [noise_type],
                        'accs': [acc]
                        })

    # Log id, graph, to overview file
    run_id = run.info['id']
    info_dict = {'algorithm': algorithm,
                 'graph': graph,
                 'noise_type': noise_type,
                 'metric': acc,
                 'run-id': run_id,
                 'load': load,
                 }

    save_config_info(save_file, info_dict)
