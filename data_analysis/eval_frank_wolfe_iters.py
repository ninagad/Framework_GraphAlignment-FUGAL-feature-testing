# Standard lib imports
import os

# Lib import
import pandas as pd

# Local import
from data_analysis.utils import get_acc_file_as_df, get_git_root
from data_analysis.test_run_configurations import (test_configuration_graph_iters_nu_mu_sinkhorn,
                                     test_run_have_expected_frank_wolfe_iters,
                                     test_runs_have_ordered_graphs,
                                     get_training_graph_names
                                     )


def average_accuracy_per_graph(runs: list[int]) -> list[float]:
    avg_accs = []
    for run in runs:
        df = get_acc_file_as_df(run)
        avg = 100 * df.values.mean()
        avg_accs.append(avg)

    return avg_accs


def get_avg_accs(grouped_runs: dict, aggregate_runs: bool) -> dict:
    avg_acc_dict = {}

    for key, runs in grouped_runs.items():
        avg_accs = average_accuracy_per_graph(runs)

        avg_acc_percent = (sum(avg_accs) / len(avg_accs))

        if aggregate_runs:
            avg_acc_dict[key] = avg_acc_percent
        else:
            avg_acc_dict[key] = avg_accs + [avg_acc_percent]

    return avg_acc_dict


def check_runs_have_correct_config(grouped_runs: dict):
    nu = 447.24
    mu = 442.66
    sinkhorn = 0.00141

    test_configuration_graph_iters_nu_mu_sinkhorn(grouped_runs, nu, mu, sinkhorn)

    for fw, runs in grouped_runs.items():
        for run in runs:
            test_run_have_expected_frank_wolfe_iters(run, fw)


def write_to_file(table, sources, filename):
    # Generate and save table
    root = get_git_root()
    path = os.path.join(root, 'tables', filename)

    with open(path, "w") as file:
        file.write(f'sources used for computation: {sources}')
        file.write('\n\n')
        file.write(table)

def generate_latex_table(content: dict, column_names: list[str], index_name: str):
    df = pd.DataFrame.from_dict(content, orient='index', columns=column_names)
    df.index.name = index_name

    # Generate LaTeX table
    table = df.to_latex(#column_format='lr',
                        float_format=f"{{:0.2f}}".format)

    return table


def process_frank_wolfe_data(res_dict: dict, filename: str, aggregate_across_graphs: bool = True):
    check_runs_have_correct_config(res_dict)

    avg_accs = get_avg_accs(res_dict, aggregate_across_graphs)

    # Convert dictionary to a DataFrame
    if aggregate_across_graphs:
        columns = ['avg. accuracy in \%']
    else:
        test_runs_have_ordered_graphs(res_dict)
        graph_names = get_training_graph_names()
        columns = graph_names + ['avg. accuracy in \%']

    table = generate_latex_table(avg_accs, columns, 'FW iterations')
    write_to_file(table, res_dict, filename)


def process_sinkhorn_0_0014():
    filename = 'sinkhorn_reg=0.0014-on-fw-graphs.txt'
    numeric_errors = {10: [12508, 12509, 12510, 12511]}

    # Check configurations
    nu = 447.24
    mu = 442.66
    sinkhorn = 0.0014
    test_configuration_graph_iters_nu_mu_sinkhorn(numeric_errors, nu, mu, sinkhorn)

    # Aggregate and generate table
    avg_accs = get_avg_accs(numeric_errors, aggregate_runs=False)
    columns = get_training_graph_names() + ['avg. accuracy in \%']
    table = generate_latex_table(avg_accs, columns, '')
    write_to_file(table, numeric_errors, filename)

if __name__ == '__main__':
    fw_dict = {1: [12450, 12451, 12452, 12453],
               2: [12413, 12414, 12415, 12416],
               3: [12454, 12455, 12456, 12457],
               4: [12429, 12430, 12431, 12432],
               6: [12433, 12434, 12435, 12436],
               8: [12437, 12438, 12439, 12440],
               10: [12445, 12446, 12447, 12461]}

    filename = 'frank_wolfe_table.txt'
    #process_frank_wolfe_data(fw_dict, filename)

    filename = 'frank_wolfe_not_aggregated.txt'
    #process_frank_wolfe_data(fw_dict, filename, aggregate_across_graphs=False)

    # To compare the best result's accuracy exactly with the best result with numeric errors.
    fw2_on_forward_selected_graphs = {2: [12486, 12487, 12488, 12489]}

    filename = 'fw2_on_third_FS_graphs.txt'
    #process_frank_wolfe_data(fw2_on_forward_selected_graphs, filename, aggregate_across_graphs=False)

    process_sinkhorn_0_0014()



