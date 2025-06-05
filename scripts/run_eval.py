import copy
import os.path

from experiment.experiments import _algs
from scripts.run_utils import run_alg, allowed_algorithms
from data_analysis.utils import get_forward_selected_features, get_git_root, get_eval_graph_run_ids, get_appendix_eval_graph_run_ids, get_15_features

from enums.scalingEnums import ScalingEnums


def run_eval_graphs(save_file: str, algorithm: allowed_algorithms, args_lst, all_algs_lst: list, acc: int = 0):
    root = get_git_root()
    path = os.path.join(root, 'overview-of-runs', save_file)

    for graph, load_id in get_eval_graph_run_ids().items():
        run_alg(path, algorithm, all_algs_lst, args_lst, graph, load_id, acc)


def run_appendix_eval_graphs(save_file: str, algorithm: allowed_algorithms, args_lst, all_algs_lst: list, acc: int = 0):
    root = get_git_root()
    path = os.path.join(root, 'overview-of-runs', save_file)

    for graph, load_id in get_appendix_eval_graph_run_ids().items():
        run_alg(path, algorithm, all_algs_lst, args_lst, graph, load_id, acc)


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
    args_lst = [
        {'features': get_forward_selected_features(),
         'nu': 447.24,
         'mu': 442.66,
         'sinkhorn_reg': 0.00141,
         'scaling': ScalingEnums.COLLECTIVE_ROBUST_NORMALIZATION,
         'frank_wolfe_iters': 2,
         }
    ]

    acc = 0

    run_eval_graphs('FUGAL-eval.txt', 'fugal', args_lst, all_algs, acc)
    run_appendix_eval_graphs('FUGAL-appendix-eval.txt', 'fugal', args_lst, algs_args, acc)

def run_pca(all_algs):
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

    acc = 0

    run_eval_graphs('PCA-eval.txt', 'fugal', args_lst, all_algs, acc)


if __name__ == '__main__':
    algs_args = copy.copy(_algs)

    run_regal(algs_args)
    run_isorank(algs_args)
    run_grampa(algs_args)
    run_proposed_fugal(algs_args)
    run_pca(algs_args)
