import argparse
import sys
import os
import numpy as np

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from plot import load_data, transform_df


def parse(run):
    """
        Calculate average accuracy accross all iterations and noise levels


        Args:
            run: indexe of plot to include in calculation


        Returns:
            the result of the calculation

        """

    # Load source and baseline
    df, _, graphs, _ = load_data(run, '', 'acc')
    if len(graphs) > 1:  # there should not be more than one graph
        raise ValueError
    graph = graphs[0]
    df.rename(columns={'Unnamed: 0': 'Features', 'Unnamed: 1': 'Noise-level'}, inplace=True)

    # Fill NaN values with the previous row values
    df['Features'] = df['Features'].ffill()

    df = df.replace(-1, np.nan)  # Replace numeric errors with NaN, so they are excluded from the mean calculation.

    df['mean'] = df.iloc[:,
                    (df.columns != 'Features') & (df.columns != 'Noise-level') & (df.columns != 'variable')].mean(axis=1)

    return df['mean'].mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--run',
                        help='The run index of the file')

    args = parser.parse_args()

    run = args.run

    # parse all graphs with min max feature normalization
    print("The average accuracy is: ", parse(run))



