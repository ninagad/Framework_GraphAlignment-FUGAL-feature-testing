import argparse
import pandas as pd
from tabulate import tabulate

import numpy as np
from plot import load_data, transform_df

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
        df, _, graphs, _ = load_data(source, '', 'acc')
        if len(graphs) > 1: # there should not be more than one graph
            raise ValueError
        graph = graphs[0]
        df.rename(columns={'Unnamed: 0': 'Features', 'Unnamed: 1': 'Noise-level'}, inplace=True)

        baseline_df, _, _, _ = load_data(baseline, '', 'acc')
        baseline_df = transform_df(baseline_df)

        # Fill NaN values with the previous row values
        df['Features'] = df['Features'].ffill()

        df = df.replace(-1, np.nan)  # Replace numeric errors with NaN, so they are excluded from the mean calculation.

        # step 1: calculate the minimum data point from each iteration
        df['min'] = df.iloc[:,
                 (df.columns != 'Features') & (df.columns != 'Noise-level') & (df.columns != 'variable')].min(axis=1)

        # step 2: calculate the average deviation from baseline for each graph across all noise levels
        nr_of_features = len(pd.unique(df['Features']))
        df['min'] = df['min'] - pd.concat([baseline_df['mean']] * nr_of_features, ignore_index=True) # Find deviation from baseline

        df = df.groupby(['Features']).mean()

        dfs.append(df)

    # step 3: determine average deviation from baseline for each feature across all graphs
    combined_df = pd.concat(dfs)
    combined_df = combined_df.groupby(['Features']).mean()

    #combined_df = combined_df[combined_df['min'] > combined_df['min'].quantile(0.75)] # only consider the top 25 % performing features

    return combined_df







if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--fn',
                        nargs='+',
                        help='The indexes of the files used for the calculations')

    parser.add_argument('--dn',
                        nargs='+',
                        help='The indexes of the files used for the calculations')

    parser.add_argument('--s',
                        nargs='+',
                        help='The indexes of the files used for the calculations')

    parser.add_argument('--rn',
                        nargs='+',
                        help='The indexes of the files used for the calculations')

    parser.add_argument('--baselines',
                        nargs='+',
                        help='The index of the file used for the baseline')

    args = parser.parse_args()
    feature_mm_norm = args.fn
    distance_mm_norm = args.dn
    feature_stand = args.s
    feature_rob_norm = args.rn
    baselines = args.baselines

    # parse all graphs with min max feature normalization
    feature_mm_norm_df = parse(feature_mm_norm, baselines)

    # parse all graphs with min max distance normalization
    distance_mm_norm_df = parse(distance_mm_norm, baselines)

    # parse all graphs with feature standardization
    feature_stand_df = parse(feature_stand, baselines)

    # parse all graphs with robust feature normalization
    feature_rob_norm_df = parse(feature_rob_norm, baselines)

    # Create new dataframe with scaling values
    scaling_df = pd.concat([feature_mm_norm_df['min'], distance_mm_norm_df['min'], feature_stand_df['min'], feature_rob_norm_df['min']], axis=1)
    scaling_df.columns =['feature_mm_norm', 'distance_mm_norm', 'feature_stand', 'feature_rob_norm']

    scaling_df = scaling_df.drop(['[KATZ_CENTRALITY]'])
    # Find the scaling method for each feature that has the maximum average distance to baseline
    scaling_df['max'] = scaling_df.idxmax(axis=1)

    print("The scaling method that has the maximum average distance to baseline for the most features is: ", scaling_df['max'].mode()[0])
    print(tabulate(scaling_df.round(4), headers='keys', tablefmt='latex_booktabs'))

