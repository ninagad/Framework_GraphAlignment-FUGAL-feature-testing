import argparse
import sys
import os
import numpy as np
import pandas as pd
import json

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

def load_data(source: int, xaxis: str, yaxis: str) -> (pd.DataFrame, int, str, int):
    """
    Loads data from excel file

    Args:
        source: index of directory to load from

    Returns:
        dataframe, mu, graph name and #iterations.

    """

    server_runs_path = os.path.join((os.path.dirname(__file__)), 'runs')
    res_dir_path = os.path.join(server_runs_path, f'{source}')
    res_path = os.path.join(res_dir_path, 'res', f'{yaxis}.xlsx')

    # Make sure the excel file is not open in Excel! Otherwise, this fails with Errno 13 permission denied.
    sheet_dict = pd.read_excel(res_path, sheet_name=None)  # sheet_name = None -> all sheets are loaded

    for sheet_name, df in sheet_dict.items():
        # Get the p value from the sheet name and add it as a new column in the dfs

        if xaxis == 'p':
            var = sheet_name.split('p=')[1]
            df['variable'] = float(var)

        elif xaxis == 'External p':
            var = sheet_name.split('extp=')[1]
            df['variable'] = float(var)

        elif xaxis == 'k':
            var = sheet_name.split('k=')[1].split('_')[0]
            df['variable'] = float(var)

        elif xaxis == 'n':
            var = sheet_name.split('n=')[1].split('_')[0]
            df['variable'] = int(var)

    df = pd.concat(list(sheet_dict.values()), axis=0, ignore_index=True)

    # Opening JSON file
    f = open(os.path.join(res_dir_path, 'config.json'))
    # returns JSON object as a dictionary
    meta_data = json.load(f)

    try:
        mu = meta_data['algs'][0][1]['mu']
    # Other algorithms than FUGAL
    except KeyError:
        mu = None

    graphs = meta_data['graph_names']
    iters = meta_data['iters']

    return df, mu, graphs, iters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--run',
                        help='The run index of the file')

    args = parser.parse_args()

    run = args.run

    # parse all graphs with min max feature normalization
    print("The average accuracy is: ", parse(run))



