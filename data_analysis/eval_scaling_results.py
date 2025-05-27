import os

import pandas as pd
from tabulate import tabulate
import numpy as np

from enums.featureEnums import FeatureExtensions
from data_analysis.utils import get_acc_file_as_df, get_git_root
from data_analysis.test_run_configurations import test_graphs_are_training_graphs, test_run_has_5_iterations, \
    test_run_has_6_noise_levels, test_run_have_expected_mu


def check_configuration(runs: list[int]):
    test_graphs_are_training_graphs(runs)

    for run in runs:
        test_run_has_5_iterations(run)
        test_run_have_expected_mu(run, 1)
        test_run_has_6_noise_levels(run)


def parse(sources):
    """
        Parse results from an experiment as follows:
        - Find the minimum data point over all iterations.
        - Compute average min acc across all noise levels and graphs.

        Args:
            sources: list of indexes of runs to include in calculation

        Returns:
            the result of the calculations

        """
    check_configuration(sources)

    dfs = []
    for source in sources:
        # Load source and baseline
        df = get_acc_file_as_df(source)
        df = df.replace(-1, np.nan)  # Replace numeric errors with NaN, so they are excluded from the mean calculation.

        # step 1: calculate the minimum data point from each iteration
        df['min'] = df.min(axis=1)

        # step 2: calculate mean of the mins over all noise levels
        df = 100 * (df.groupby(level='Feature')['min'].mean())

        dfs.append(df)

    # step 3: determine average acc for each feature across all graphs
    combined_df = pd.concat(dfs)
    combined_df = combined_df.groupby(level='Feature').mean()

    return combined_df


if __name__ == "__main__":
    # First four = original features, last four augmented clustering and deg
    scaling_dict = {'individual_mm_norm': [377, 378, 379, 380,
                                           804, 805, 807, 808],

                    'individual_stand': [385, 386, 387, 388,
                                         1021, 1022, 1023, 1024],

                    'individual_rob_norm': [381, 382, 383, 384,
                                            833, 834, 835, 836],

                    'collective_mm_norm': [365, 366, 367, 368,
                                           1009, 1010, 1011, 1012],

                    'collective_stand': [373, 374, 375, 376,
                                         1017, 1018, 1019, 1020],

                    'collective_rob_norm': [369, 370, 371, 372,
                                            1025, 1026, 1027, 1028]
                    }

    dfs = []
    for scaling, run_indices in scaling_dict.items():
        df1 = parse(run_indices[:4])
        df2 = parse(run_indices[4:])

        df = pd.concat([df1, df2])
        df = df.rename(scaling)
        dfs.append(df)

    scaling_df = pd.concat(dfs, axis=1)

    # Remove Katz centrality if it is there
    drop_indices = ['[KATZ_CENTRALITY]', '[VAR_EGO_CLUSTER]']
    for idx in drop_indices:
        try:
            scaling_df = scaling_df.drop(index=[idx])
        except KeyError:
            pass

    k = scaling_df.shape[0] // 4  # Get 25 percent of #features
    scaling_df = pd.concat([scaling_df.nlargest(k, columns=[col]) for col in scaling_df.columns])
    scaling_df = scaling_df.drop_duplicates()

    # Find the scaling method for each feature that has the maximum average distance to baseline
    scaling_df['max'] = scaling_df.idxmax(axis=1)

    # Convert feature names to labels
    FE = FeatureExtensions()
    scaling_df.index = scaling_df.index.map(lambda x: FE.transform_feature_str_to_label(x))
    scaling_df = scaling_df.sort_index()

    latex_table = tabulate(scaling_df.round(2), headers='keys', tablefmt='latex_booktabs')

    # Write to file
    root = get_git_root()
    path = os.path.join(root, 'tables', "scaling-table.txt")
    with open(path, "w") as file:
        file.write(str(scaling_dict))
        file.write('\n\n')
        file.write(f"The scaling method that has the maximum average accuracy "
                   f"for the most features is: {scaling_df['max'].mode()[0]}"
                   f"\n\n")

        file.write(latex_table)
