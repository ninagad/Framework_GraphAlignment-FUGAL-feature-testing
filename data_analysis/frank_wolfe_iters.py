# Standard lib imports
import os

# Lib import
import pandas as pd

# Local import
from utils import get_acc_file_as_df, get_git_root
from test_run_configurations import (test_graphs_are_training_graphs,
                                     test_run_has_6_noise_levels,
                                     test_run_have_expected_nu_and_mu,
                                     test_run_have_expected_frank_wolfe_iters,
                                     test_run_have_expected_sinkhorn_reg)


def average_accuracy_across_graphs(runs: list[int]) -> float:
    avg_accs = []
    for run in runs:
        df = get_acc_file_as_df(run)
        avg = df.values.mean()
        avg_accs.append(avg)

    avg_acc_percent = 100 * (sum(avg_accs) / len(avg_accs))

    return avg_acc_percent


def get_avg_accs(grouped_runs: dict) -> dict:
    avg_acc_dict = {}

    for key, runs in grouped_runs.items():
        avg_acc = average_accuracy_across_graphs(runs)

        avg_acc_dict[key] = avg_acc

    return avg_acc_dict


def check_runs_have_correct_config(grouped_runs: dict):
    nu = 447.24
    mu = 442.66
    sinkhorn = 0.00141

    for fw, runs in grouped_runs.items():
        test_graphs_are_training_graphs(runs)

        for run in runs:
            test_run_has_6_noise_levels(run)
            test_run_have_expected_nu_and_mu(run, nu, mu)
            test_run_have_expected_frank_wolfe_iters(run, fw)
            test_run_have_expected_sinkhorn_reg(run, sinkhorn)


def process_frank_wolfe_data():
    fw_dict = {1: [12450, 12451, 12452, 12453],
               2: [12413, 12414, 12415, 12416],
               3: [12454, 12455, 12456, 12457],
               4: [12429, 12430, 12431, 12432],
               6: [12433, 12434, 12435, 12436],
               8: [12437, 12438, 12439, 12440],
               10: [12445, 12446, 12447, 12461]}

    check_runs_have_correct_config(fw_dict)

    avg_accs = get_avg_accs(fw_dict)

    # Convert dictionary to a DataFrame
    df = pd.DataFrame(avg_accs.items(), columns=['FW iterations', 'avg. accuracy in \%'])
    # df = df.round({'avg. accuracy in \%': 2})

    # Generate and save table
    filename = 'frank_wolfe_table.txt'
    root = get_git_root()
    path = os.path.join(root, 'tables', filename)

    # Generate LaTeX table
    df.to_latex(index=False, column_format='lr',
                float_format=f"{{:0.2f}}".format, buf=path)


if __name__ == '__main__':
    process_frank_wolfe_data()
