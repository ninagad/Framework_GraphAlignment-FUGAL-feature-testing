from typing import Literal

import workexp
from experiment.experiments import alggs as get_run_list, _algs, get_graph_paths as get_graph_paths

allowed_algorithms = Literal['fugal', 'cugal', 'isorank', 'regal', 'grampa', 'cone']


def get_algo_id(algorithm: allowed_algorithms):
    algo_id_dict = {'fugal': 12,
                    'cugal': 22,
                    'isorank': 6,
                    'regal': 3,
                    'grampa': 20,
                    'cone': 1}

    return algo_id_dict[algorithm]


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
    _algs[:] = all_algs

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
                 'run-id': run_id,
                 'load': load,
                 }

    save_config_info(save_file, info_dict)
