import argparse

import matplotlib.pyplot as plt
import pandas as pd
import os
import json

from feature import FeatureExtensions as FE
from plot_utils import PlotUtils as PU
import numpy as np

def load_data(source: int, plottype: str) -> (pd.DataFrame, int, str, int):
    """
    Loads data from excel file

    Args:
        source: index of directory to load from

    Returns:
        dataframe, mu, graph name and #iterations.

    """

    server_runs_path = os.path.join((os.path.dirname(__file__)), 'Server-runs')
    res_dir_path = os.path.join(server_runs_path, f'{source}')

    res_path = os.path.join(res_dir_path, 'res', 'acc.xlsx')

    # Make sure the excel file is not open in Excel! Otherwise, this fails with Errno 13 permission denied.
    sheet_dict = pd.read_excel(res_path, sheet_name=None)  # sheet_name = None -> all sheets are loaded

    for sheet_name, df in sheet_dict.items():
        # Get the p value from the sheet name and add it as a new column in the dfs

        if plottype == 'p':
            var = sheet_name.split('p=')[1]
            df['variable'] = float(var)

        elif plottype == 'External p':
            var = sheet_name.split('extp=')[1]
            df['variable'] = float(var)

        elif plottype == 'k':
            var = sheet_name.split('k=')[1].split('_')[0]
            df['variable'] = float(var)

        elif plottype == 'n':
            var = sheet_name.split('n=')[1].split('_')[0]
            df['variable'] = int(var)

    df = pd.concat(list(sheet_dict.values()), axis=0, ignore_index=True)

    # Opening JSON file
    f = open(os.path.join(res_dir_path, 'config.json'))
    # returns JSON object as a dictionary
    meta_data = json.load(f)

    mu = meta_data['algs'][0][1]['mu']
    graphs = meta_data['graph_names']
    iters = meta_data['iters']

    return df, mu, graphs, iters


def transform_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns, fills NaN values and computes mean over all iterations

    Args:
        df: dataframe to transform
        plottype: either 'noise' or 'density'

    Returns:
        transformed dataframe

    """
    df.rename(columns={'Unnamed: 0': 'Features', 'Unnamed: 1': 'Noise-level'}, inplace=True)

    # Fill NaN values with the previous row values
    df['Features'] = df['Features'].ffill()

    df = df.replace(-1, np.nan)  # Replace numeric errors with NaN, so they are excluded from the mean calculation.

    df['mean'] = df.iloc[:, (df.columns != 'Features') & (df.columns != 'Noise-level') & (df.columns != 'variable')].mean(axis=1)

    return df


def get_color_marker_label(feature: str):
    """

    Args:
        feature:

    Returns:

    """
    pu = PU()

    if ',' not in feature:  # It is a single feature
        feature = FE.to_feature(feature)

        # to_feature returns None, if the feature is not an ENUM option.
        if feature is None:
            return None, None, None

        color = pu.to_color(feature)
        marker = pu.to_marker(feature)

        label = FE.to_label(feature)

    else:  # It is a combination of features
        features_in_combination = feature.replace(' ', '').split(',')
        features = [FE.to_feature(name) for name in features_in_combination]

        color = pu.to_colors(features)
        marker = pu.to_markers(features)
        label = FE.to_labels(features)

    return color, marker, label


def plot(plottype: str, baseline: int, source: int, title: str, outputdir: str):
    """

    Args:
        plottype:
        baseline:
        sources:
        title:
        outputdir:

    Returns:

    """
    df, mu, graphs, iters = load_data(source, plottype)
    df = transform_df(df)

    xname = PU().to_column_name(plottype)

    # Create plot
    plt.figure(figsize=(10, 6))

    # Loop through unique features and plot each one
    # Define colorscale for this set of features
    features = df['Features'].unique()

    for i, feature in enumerate(features):

        subset = df[df['Features'] == feature]

        color, marker, label = get_color_marker_label(feature)

        # Don't plot feature if it is not an ENUM feature
        if label is None:
            continue

        plt.plot(subset[xname], subset['mean'], color=color, marker=marker, label=label)

    # Draw baseline
    if baseline is not None:
        # Load baseline df
        baseline_df, b_mu, b_graphs, b_iters = load_data(baseline, plottype)
        baseline_df = transform_df(baseline_df)

        # Check that graph name and #iterations match source
        if (graphs[0] != b_graphs[0]) or (iters != b_iters):
            raise Exception(
                f"The baseline does not have the same meta data graphs: {b_graphs}, iterations: {b_iters} as the main source graphs:{graphs}, iterations: {iters} :(")

        # Baseline trace color (red)
        baseline_color = plt.get_cmap("Set1")(0)

        label = 'baseline (no features)'

        plt.plot(baseline_df[xname], baseline_df['mean'], color=baseline_color, label=label)

    # Layout plot
    plt.ylim(-0.1, 1.1)

    plt.xlabel(plottype)

    plt.ylabel('Accuracy')

    plt.suptitle(title, fontsize=24, x=0.40, y=0.97)

    graph = graphs[0]
    # Format graph info in subtitle
    if plottype == 'Noise-level':
        graph_info = graph
    else:
        if plottype == 'p':  # NWS graph
            remove_val = '_p='
            graph = graph.replace('_str', 's') # Reformat graph name nw_str -> nws
        elif plottype == 'External p':  # Stochastic block model
            remove_val = '_extp='
        elif plottype == 'k':
            remove_val = '_k='
            graph = graph.replace('_str', 's')  # Reformat graph name nw_str -> nws
        elif plottype == 'n':
            remove_val = '_n='
            graph = graph.replace('_str', 's')  # Reformat graph name nw_str -> nws

            k_0 = graphs[0].split('_k=')[1].split('_')[0]
            k_1 = graphs[1].split('_k=')[1].split('_')[0]

            if k_0 != k_1:
                n_0 = graphs[0].split('_n=')[1].split('_')[0]
                k = float(n_0) / float(k_0)
                graph = graph.replace('_k='+str(k_0), '_k='+str(int(k))+'divn')
        else:
            remove_val = None

        info_split = graph.split(remove_val)
        graph = info_split[0] + ('_' + info_split[1].split('_')[1] if '_' in info_split[1] else '')  # Remove variable value

        graph_info = (graph
                      .replace('_', ', ')
                      .replace('=', ': ')  # Reformat = -> :
                      .replace('div', '/'))
        graph_info += ', noise-level: ' + str(df.at[0, 'Noise-level'])  #  Add noise-level


    plt.title(label=f'$\mu$: {mu}, graph: {graph_info}', fontsize=12)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', title='Features')
    plt.tight_layout()
    plt.grid(True)

    # Save plot
    path = os.path.join(os.path.dirname(__file__), 'plots', outputdir, f'{graph}-mu={mu}.svg')
    plt.savefig(path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--plottype',
                        choices=['External p', 'p', 'k', 'Noise-level', 'n'],
                        default='Noise-level')

    parser.add_argument('--baseline',
                        type=int,
                        default=None,
                        help='The index of the file used for the baseline')

    parser.add_argument('--source',
                        type=int,
                        help='The index of the file used for the main plot')

    #parser.add_argument('--sources', help='The index/indices of the file(s) used for the main plot',
                        #type=lambda s: [int(item) for item in s.split(',')])

    parser.add_argument('--title',
                        type=str,
                        default='Ablation study for FUGAL features',
                        help='The title of the plot, remember to use " " ')

    parser.add_argument('--outputdir',
                        type=str,
                        default='',
                        help='The directory the plot should be saved in')

    args = parser.parse_args()

    plottype = args.plottype

    baseline = args.baseline
    source = args.source
    title = args.title
    outputdir = args.outputdir

    plot(plottype=plottype, baseline=baseline, source=source, title=title, outputdir=outputdir)
