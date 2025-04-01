import argparse
import pandas as pd
from tabulate import tabulate
import sys
import os
import numpy as np

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
        if len(graphs) > 1:  # there should not be more than one graph
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
        df['min'] = df['min'] - pd.concat([baseline_df['mean']] * nr_of_features,
                                          ignore_index=True)  # Find deviation from baseline

        df = df.groupby(['Features']).mean()

        dfs.append(df)

    # step 3: determine average deviation from baseline for each feature across all graphs
    combined_df = pd.concat(dfs)
    combined_df = combined_df.groupby(['Features']).mean()

    # combined_df = combined_df[combined_df['min'] > combined_df['min'].quantile(0.75)] # only consider the top 25 % performing features

    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--ifn',
                        nargs='+',
                        help='The indexes of the files containing the individual feature normalization')

    parser.add_argument('--ifs',
                        nargs='+',
                        help='The indexes of the files containing the individual feature standardization')

    parser.add_argument('--irfn',
                        nargs='+',
                        help='The indexes of the files containing the individual robust feature normalization')

    parser.add_argument('--cfn',
                        nargs='+',
                        help='The indexes of the files containing the collective feature normalization')

    parser.add_argument('--cfs',
                        nargs='+',
                        help='The indexes of the files containing the collective feature standardization')

    parser.add_argument('--crfn',
                        nargs='+',
                        help='The indexes of the files containing the collective robust feature normalization')

    parser.add_argument('--baselines',
                        nargs='+',
                        help='The index of the file used for the baseline')

    args = parser.parse_args()

    # Scaling computed over each graph separately
    individual_mm_norm = args.ifn
    individual_stand = args.ifs
    individual_rob_norm = args.irfn

    # Scaling computed over both graphs
    collective_mm_norm = args.cfn
    collective_stand = args.cfs
    collective_rob_norm = args.crfn

    baselines = args.baselines

    # parse all graphs with min max feature normalization
    individual_mm_norm_df = parse(individual_mm_norm, baselines)

    # parse all graphs with feature standardization
    individual_stand_df = parse(individual_stand, baselines)

    # parse all graphs with robust feature normalization
    individual_rob_norm_df = parse(individual_rob_norm, baselines)

    #
    # parse all graphs with min max feature normalization
    collective_mm_norm_df = parse(collective_mm_norm, baselines)

    # parse all graphs with feature standardization
    collective_stand_df = parse(collective_stand, baselines)

    # parse all graphs with robust feature normalization
    collective_rob_norm_df = parse(collective_rob_norm, baselines)

    #
    # Create new dataframe with scaling values
    scaling_df = pd.concat([individual_mm_norm_df['min'],
                            individual_stand_df['min'],
                            individual_rob_norm_df['min'],
                            collective_mm_norm_df['min'],
                            collective_stand_df['min'],
                            collective_rob_norm_df['min']],
                           axis=1)

    scaling_df.columns = ['individual_mm_norm',
                          'individual_stand',
                          'individual_rob_norm',
                          'collective_mm_norm',
                          'collective_stand',
                          'collective_rob_norm']

    # Remove Katz centrality if it is there
    try:
        scaling_df = scaling_df.drop('[KATZ_CENTRALITY]')
    except KeyError:
        pass

    # Find the scaling method for each feature that has the maximum average distance to baseline
    scaling_df['max'] = scaling_df.idxmax(axis=1)

    f = open("myfile.txt", "w")
    print("The scaling method that has the maximum average distance to baseline for the most features is: ",
          scaling_df['max'].mode()[0])

    # Print the union of the top 25 performing features
    rows = ['[AVG_EGO_DEG]', '[DEGREE_CENTRALITY]', '[DEG]', '[EGO_EDGES]', '[EGO_NEIGHBORS]',
            '[PAGERANK]', '[MEDIAN_EGO_DEGS]', '[EGO_OUT_EDGES]', '[MAX_EGO_DEGS]', '[RANGE_EGO_DEGS]']
    print(scaling_df.loc[rows])

    latex_table = tabulate(scaling_df.loc[rows].round(4), headers='keys', tablefmt='latex_booktabs')
    print(latex_table)

    # Write to file
    file_path = os.path.join('..', 'tables', "scaling-table.txt")
    with open(file_path, "w") as file:
        args_dict = vars(args)

        file.write(str(args_dict))
        file.write('\n\n')
        file.write(latex_table)




