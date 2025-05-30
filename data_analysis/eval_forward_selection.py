import os

import pandas as pd

from enums.featureEnums import FeatureExtensions
from data_analysis.utils import get_parameter, get_acc_file_as_df, get_git_root
from data_analysis.test_run_configurations import test_configuration_graph_iters_nu_mu_sinkhorn


def top_performing_feature(sources: list[int]):
    first_run = sources[0]
    nu = get_parameter(first_run, 'nu')
    mu = get_parameter(first_run, 'mu')
    sinkhorn_reg = get_parameter(first_run, 'sinkhorn_reg')

    dfs = []

    for source in sources:
        graph_df = get_acc_file_as_df(source)

        # Remove Katz centrality if it is there
        try:
            graph_df = graph_df.drop('[KATZ_CENTRALITY]', level=0)
        except KeyError:
            pass

        # For each feature+noise combination compute the mean over the iterations
        graph_df = graph_df.mean(axis=1).rename('avg accuracy')

        dfs.append(graph_df)

    # Stack the dfs from each graph
    df = pd.concat(dfs, axis=0)

    # Compute the mean over the different graphs and noise levels for each feature
    df = df.groupby(level=[0]).mean()
    df = df.sort_values(ascending=False)
    df.index.name = 'Feature'

    max_feature = df.idxmax()

    df = df.reset_index()

    # Convert feature names to labels
    fe = FeatureExtensions()
    df['Feature'] = df['Feature'].apply(lambda x: fe.transform_feature_str_to_label(x))

    return df, max_feature, nu, mu, sinkhorn_reg


def save_to_file(df, feature, sources, nu, mu, reg, round_no):
    latex_table = pd.Series.to_latex(df, index=False)

    # Write to file
    root = get_git_root()
    file_path = os.path.join(root, 'tables', f"feature-forward-selection-{round_no}-features.txt")
    with open(file_path, "w") as file:
        file.write(f'sources used for computation: {sources}')
        file.write('\n\n')
        file.write(f'nu: {nu} \n'
                   f'mu: {mu} \n'
                   f'sinkhorn_reg: {reg} \n \n')

        file.write(f'Best feature: {feature}')
        file.write('\n\n')
        file.write(latex_table)


def forward_feature_selection():
    sources_dict = {1: [7522, 7523, 7524, 7525],
                    # 2: [11427, 11428, 11429, 11430],
                    # 2: [115, 116, 117, 118],  # Skadi
                    2: [12384, 12385, 12386, 12387],
                    # 3: [12380, 12381, 12382, 12383],
                    3: [12388, 12389, 12390, 12391],
                    4: [12400, 12401, 12402, 12403],
                    5: [17351, 17352, 17353, 17350]
                    }

    # Test configurations of all runs
    nu_initial, mu_initial = 494, 125.98
    sinkhorn = 0.0014
    nu_remaining, mu_remaining = 447.24, 442.66

    # Test that all runs have the correct graphs, nu, mu, reg and iters
    test_configuration_graph_iters_nu_mu_sinkhorn({1: sources_dict[1]}, nu_initial, mu_initial, sinkhorn)
    test_configuration_graph_iters_nu_mu_sinkhorn({k: v for k, v in sources_dict.items() if k != 1}, nu_remaining,
                                                  mu_remaining, sinkhorn)

    for round_no, sources in sources_dict.items():
        df, feature, nu, mu, reg = top_performing_feature(sources)
        save_to_file(df, feature, sources, nu, mu, reg, round_no)


if __name__ == "__main__":
    forward_feature_selection()
