import copy
import os.path

from experiment.experiments import _algs
from scripts.run_utils import run_alg, allowed_algorithms, eval_graphs, \
    get_proposed_fugal_w_fixed_feature_set_arguments, get_proposed_fugal_w_pca_arguments, \
    get_original_fugal_eval_graph_mus, get_original_fugal_args
from data_analysis.utils import get_forward_selected_features, get_git_root, get_eval_graph_run_ids, \
    get_appendix_eval_graph_run_ids, get_15_features, get_metric_noise_graph_run_ids

from enums.scalingEnums import ScalingEnums


def run_eval_graphs(save_file: str, algorithm: allowed_algorithms, args_lst, all_algs_lst: list):
    root = get_git_root()
    path = os.path.join(root, 'overview-of-runs', save_file)

    for graph, load_id in get_eval_graph_run_ids().items():
        run_alg(path, algorithm, all_algs_lst, args_lst, graph, load_id)


def run_eval_graph(graph: eval_graphs, save_file: str, algorithm: allowed_algorithms, args_lst, all_algs_lst: list):
    root = get_git_root()
    path = os.path.join(root, 'overview-of-runs', save_file)

    load_id = get_eval_graph_run_ids()[graph]
    run_alg(path, algorithm, all_algs_lst, args_lst, graph, load_id)


def run_appendix_eval_graphs(save_file: str, algorithm: allowed_algorithms, args_lst, all_algs_lst: list):
    root = get_git_root()
    path = os.path.join(root, 'overview-of-runs', save_file)

    for graph, load_id in get_appendix_eval_graph_run_ids().items():
        run_alg(path, algorithm, all_algs_lst, args_lst, graph, load_id)


def run_metric_noise_graphs(save_file: str, algorithm: allowed_algorithms, args_lst, all_algs_lst: list):
    root = get_git_root()
    path = os.path.join(root, 'overview-of-runs', save_file)

    metrics = [3, 5, 6]  # s3, mnc, frob
    noise_types = [1, 2, 3]

    for graph, load_id in get_metric_noise_graph_run_ids().items():
        for metric in metrics:
            for noise_type in noise_types:
                run_alg(path, algorithm, all_algs_lst, args_lst, graph, load_id, noise_type=noise_type, acc=metric)


def run_grampa(all_algs):
    args_lst = [
        {'features': get_forward_selected_features(),
         'eta': 0.2
         }
    ]

    run_eval_graphs('GRAMPA-eval.txt', 'grampa', args_lst, all_algs)


def run_regal(all_algs):
    args_lst = [
        {'features': get_forward_selected_features(),
         'gammaattr': 0.06,
         'attributes': 1,  # Use features
         'scaling': ScalingEnums.COLLECTIVE_ROBUST_NORMALIZATION
         }
    ]

    run_eval_graphs('REGAL-eval.txt', 'regal', args_lst, all_algs)


def run_isorank(all_algs):
    args_lst = [
        {'features': get_forward_selected_features(),
         'alpha': 0.89
         }
    ]

    run_eval_graphs('IsoRank-eval.txt', 'isorank', args_lst, all_algs)


def run_proposed_fugal(all_algs):
    args = get_proposed_fugal_w_fixed_feature_set_arguments()

    run_eval_graphs('FUGAL-eval.txt', 'fugal', args, all_algs)
    run_appendix_eval_graphs('FUGAL-appendix-eval.txt', 'fugal', args, algs_args)
    run_metric_noise_graphs('FUGAL-fixed-metric-noise-eval.txt', 'fugal', args, all_algs)


def run_pca(all_algs):
    args = get_proposed_fugal_w_pca_arguments()

    run_eval_graphs('PCA-eval.txt', 'fugal', args, all_algs)
    run_metric_noise_graphs('FUGAL-pca-metric-noise-eval.txt', 'fugal', args, all_algs)


def run_original_fugal(all_algs, save_file: str):
    mus = get_original_fugal_eval_graph_mus()

    for graph, mu in mus.items():
        args = get_original_fugal_args(mu)
        run_eval_graph(graph, save_file, 'fugal', args, all_algs)


def run_fugal_time_experiment(all_algs):
    for graph, _ in get_eval_graph_run_ids().items():
        # Proposed FUGAL with fixed feature set
        args = get_proposed_fugal_w_fixed_feature_set_arguments()
        run_eval_graph(graph, 'time-FUGAL-fixed-features.txt', 'fugal', args, all_algs)

        # Proposed FUGAL with PCA
        args = get_proposed_fugal_w_pca_arguments()
        run_eval_graph(graph,'time-FUGAL-pca.txt', 'fugal', args, all_algs)

        # Original FUGAL
        mu = get_original_fugal_eval_graph_mus()[graph]
        args = get_original_fugal_args(mu)
        run_eval_graph(graph, 'time-FUGAL-original.txt', 'fugal', args, all_algs)


if __name__ == '__main__':
    algs_args = copy.copy(_algs)

    run_regal(algs_args)
    run_isorank(algs_args)
    run_grampa(algs_args)
    run_proposed_fugal(algs_args)
    run_pca(algs_args)
    run_fugal_time_experiment(algs_args)
