import copy
from threading import Thread

import workexp
from experiment.experiments import alggs as get_run_list, _algs, rgraphs as get_graph_paths
from enums.scalingEnums import ScalingEnums
from enums.featureEnums import FeatureEnums
from enums.graphEnums import GraphEnums

single_features = [[FeatureEnums.DEG], [FeatureEnums.CLUSTER], [FeatureEnums.AVG_EGO_DEG],
                   [FeatureEnums.AVG_EGO_CLUSTER],
                   [FeatureEnums.EGO_EDGES], [FeatureEnums.EGO_OUT_EDGES], [FeatureEnums.EGO_NEIGHBORS],  # NetSimile
                   # Statistical features on degrees
                   [FeatureEnums.SUM_EGO_DEG], [FeatureEnums.STD_EGO_DEG], [FeatureEnums.MODE_EGO_DEGS],
                   [FeatureEnums.MEDIAN_EGO_DEGS], [FeatureEnums.MIN_EGO_DEGS], [FeatureEnums.MAX_EGO_DEGS],
                   [FeatureEnums.RANGE_EGO_DEGS], [FeatureEnums.SKEWNESS_EGO_DEGS], [FeatureEnums.KURTOSIS_EGO_DEGS],
                   # Augmented clustering features
                   [FeatureEnums.SUM_EGO_CLUSTER], [FeatureEnums.STD_EGO_CLUSTER], [FeatureEnums.RANGE_EGO_CLUSTER],
                   [FeatureEnums.MIN_EGO_CLUSTER], [FeatureEnums.MAX_EGO_CLUSTER], [FeatureEnums.MEDIAN_EGO_CLUSTER],
                   [FeatureEnums.SKEWNESS_EGO_CLUSTER], [FeatureEnums.KURTOSIS_EGO_CLUSTER],
                   # Other features
                   [FeatureEnums.ASSORTATIVITY_EGO], [FeatureEnums.INTERNAL_FRAC_EGO],
                   # Centrality measures
                   [FeatureEnums.CLOSENESS_CENTRALITY], [FeatureEnums.DEGREE_CENTRALITY],
                   [FeatureEnums.EIGENVECTOR_CENTRALITY], [FeatureEnums.PAGERANK],  # [Feature.KATZ_CENTRALITY],
                   ]

def run_FUGAL(all_algs: list, algo_args: list[dict], graphs: list[tuple], algo_id: int = 12):
    """
    Runs FUGAL (default) for the given graphs and baselines with the argument list given by algo_args.

    Args:
        all_algs: list of all default algorithm arguments
        algo_args: list of dictionaries with arguments to algorithm
        graphs: tuples of (graph, baseline)
        algo_id: id of algorithm. Default is FUGAL (12).
    """
    noises = [0, 0.05, 0.10, 0.15, 0.20, 0.25]

    iterations = 5

    run_lst = list(range(len(algo_args)))

    # Ensure _algs contains all algorithms before selecting algorithm
    _algs[:] = all_algs
    _algs[:] = get_run_list([algo_id, algo_args])

    for graph, baseline in graphs:
        graph_wrapper = get_graph_paths([graph])

        if baseline is None:
            load = []
        else:
            load = [baseline, baseline]

        workexp.ex.run(
            config_updates={'noises': noises,
                            'run': run_lst,
                            'graph_names': [graph],
                            'load': load,
                            'graphs': graph_wrapper,
                            'iters': iterations,
                            'xlabel': 'Noise-level',
                            # 'verbose': True
                            })


def run_parallel(all_algs: list, graph_dict: dict, args_lst: list[dict]):
    """
    Runs the algorithm with the args_lst for each graph in graph_dict in parallel.

    Args:
        all_algs: list of all default algorithm arguments
        graph_dict: dictionary containing tuples of (graph, baseline)
        args_lst: list of dictionaries with arguments to algorithm
    """

    threads = []
    for graph, baseline in graph_dict:
        run_args = all_algs, args_lst, [(graph, baseline)]

        thread = Thread(target=run_FUGAL, args=run_args)
        thread.start()
        threads.append(thread)

    # wait for all threads to complete before terminating
    for thread in threads:
        thread.join()


def scaling_test(all_algs: list):
    """
    Runs all types of scaling for all single features and 4 graphs
    (bio-celegans, ca-netscience, voles and euroroad).
    Args:
        all_algs: list of all default algorithm arguments
    """
    graph_dict = {GraphEnums.BIO_CELEGANS: ("bio-celegans", 36),
                  GraphEnums.CA_NETSCIENCE: ("ca-netscience", 19),
                  GraphEnums.VOLES: ("mammalia-voles-plj-trapping_100", 60),
                  GraphEnums.INF_EUROROAD: ("inf-euroroad", 20),
                  }

    for scaling in ScalingEnums:
        if scaling == ScalingEnums.NO_SCALING:
            continue

        args_lst = [
            {'features': feature_set,
             'nu': None,
             'mu': 1,
             'sinkhorn_reg': 1,
             'scaling': scaling,
             } for feature_set in single_features
        ]

        run_parallel(all_algs, graph_dict, args_lst)

def main():
    all_algs = copy.copy(_algs)

    scaling_test(all_algs)


if __name__ == "__main__":
    main()
