import argparse
import pandas as pd
from tabulate import tabulate
import sys
import os
import numpy as np

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from enums.featureEnums import FeatureExtensions


def load_data(source: int):
    server_runs_path = os.path.join((os.path.dirname(__file__)), '..', 'Server-runs')

    path = os.path.join(server_runs_path, f'{source}', 'res', 'acc.xlsx')

    # Make sure the excel file is not open in Excel! Otherwise, this fails with Errno 13 permission denied.
    df = pd.read_excel(path, index_col=[0, 1])
    df.index.names = ['Feature', 'noise']

    return df


def parse(sources, baselines):
    """
        Parse results from an experiment as follows:
        - Find the minimum data point over all iterations.
        - Compute average deviation from baseline.
        - Compute average deviation from baseline across all noise levels.
        - Return average deviation from baseline for each feature across all graphs.


        Args:
            sources: list of indexes of plots to include in calculation
            baselines: indexes of plots used as baselines. Must have same length as sources.
                       Each entry contains the baseline for the source with the corresponding entry.


        Returns:
            the result of the calculations

        """

    if len(sources) != len(baselines):
        raise ValueError

    dfs = []

    for source, baseline in zip(sources, baselines):
        # Load source and baseline
        df = load_data(source)
        df = df.replace(-1, np.nan)  # Replace numeric errors with NaN, so they are excluded from the mean calculation.

        baseline_df = load_data(baseline)
        baseline_df = baseline_df.replace(-1, np.nan)
        baseline_df['avg accuracy'] = baseline_df.mean(axis=1)

        # step 1: calculate the minimum data point from each iteration
        df['min'] = df.min(axis=1)

        # step 2: calculate the average deviation from baseline for each graph across all noise levels
        nr_of_features = df.index.get_level_values('Feature').nunique()

        baseline_dup = pd.concat([pd.Series(baseline_df['avg accuracy'].reset_index(drop=True))] * nr_of_features,
                                 ignore_index=True)

        # Find deviation from baseline
        df['min'] = pd.Series(
            df['min'].to_numpy() - baseline_dup.to_numpy(),
            index=df.index
        )

        df = df.groupby(level='Feature')['min'].mean()

        dfs.append(df)

    # step 3: determine average deviation from baseline for each feature across all graphs
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

    baselines = [36, 19, 20, 60]

    dfs = []
    for scaling, run_indices in scaling_dict.items():
        df1 = parse(run_indices[:4], baselines)
        df2 = parse(run_indices[4:], baselines)

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

    latex_table = tabulate(scaling_df.round(4), headers='keys', tablefmt='latex_booktabs')

    # Write to file
    file_path = os.path.join('..', 'tables', "scaling-table.txt")
    with open(file_path, "w") as file:
        file.write(f'Baselines: {baselines} \n')
        file.write(str(scaling_dict))
        file.write('\n\n')
        file.write(f"The scaling method that has the maximum average distance "
                   f"to baseline for the most features is: {scaling_df['max'].mode()[0]}"
                   f"\n\n")

        file.write(latex_table)
