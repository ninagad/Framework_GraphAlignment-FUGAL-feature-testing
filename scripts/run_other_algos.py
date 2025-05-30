import copy
import os.path

from experiment.experiments import _algs
from scripts.run_utils import run_alg, allowed_algorithms
from data_analysis.utils import get_forward_selected_features, get_git_root, get_eval_graph_run_ids

from enums.scalingEnums import ScalingEnums


def run_eval_graphs(save_file: str, algorithm: allowed_algorithms, args_lst):
    all_algs = copy.copy(_algs)
    root = get_git_root()
    path = os.path.join(root, 'overview-of-runs', save_file)

    for graph, load_id in get_eval_graph_run_ids().items():
        run_alg(path, algorithm, all_algs, args_lst, graph, load_id)


def run_grampa():
    args_lst = [
        {'features': get_forward_selected_features(),
         'eta': 0.2
         }
    ]

    run_eval_graphs('GRAMPA-eval.txt', 'grampa', args_lst)


def run_regal():
    args_lst = [
        {'features': get_forward_selected_features(),
         'gammaattr': 0.06,
         'attributes': 1,  # Use features
         'scaling': ScalingEnums.COLLECTIVE_ROBUST_NORMALIZATION
         }
    ]

    run_eval_graphs('REGAL-eval.txt', 'regal', args_lst)


def run_isorank():
    args_lst = [
        {'features': get_forward_selected_features(),
         'alpha': 0.88
         }
    ]

    run_eval_graphs('IsoRank-eval.txt', 'isorank', args_lst)


if __name__ == '__main__':
    run_regal()
