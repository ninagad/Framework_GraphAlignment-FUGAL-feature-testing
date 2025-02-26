import argparse

import matplotlib.pyplot as plt
import pandas as pd
import os
import json

from feature import FeatureExtensions as FE
from plot_utils import PlotUtils as PU


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
            p = sheet_name.split('p=')[1]
            df['p'] = float(p)

        elif plottype == 'External p':
            p = sheet_name.split('extp=')[1]
            df['p'] = float(p)


    df = pd.concat(list(sheet_dict.values()), axis=0, ignore_index=True)

    # Opening JSON file
    f = open(os.path.join(res_dir_path, 'config.json'))
    # returns JSON object as a dictionary
    meta_data = json.load(f)

    mu = meta_data['algs'][0][1]['mu']
    graph = meta_data['graph_names'][0]
    iters = meta_data['iters']

    return df, mu, graph, iters


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

    df['mean'] = df.iloc[:, (df.columns != 'Features') & (df.columns != 'Noise-level') & (df.columns != 'p')].mean(axis=1)

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
    df, mu, graph, iters = load_data(source, plottype)
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

        plt.plot(subset[xname], subset['mean'], color=color, marker=marker, label=label)

    # Draw baseline
    if baseline is not None:
        # Load baseline df
        baseline_df, b_mu, b_graph, b_iters = load_data(baseline, plottype)
        baseline_df = transform_df(baseline_df)

        # Check that graph name and #iterations match source
        if (graph != b_graph) or (iters != b_iters):
            raise Exception(
                f"The baseline does not have the same meta data {b_graph, b_iters} as the main source {graph, iters} :(")

        # Baseline trace color (red)
        baseline_color = plt.get_cmap("Set1")(0)

        label = 'baseline (no features)'

        plt.plot(baseline_df[xname], baseline_df['mean'], color=baseline_color, label=label)

    # Layout plot
    plt.ylim(-0.1, 1.1)

    plt.xlabel(plottype)

    plt.ylabel('Accuracy')

    plt.suptitle(title, fontsize=24, x=0.40, y=0.97)

    # Format graph info in subtitle
    if plottype == 'Noise-level':
        graph_info = graph
    else:
        if plottype == 'p':  # NWS graph
            split_val = '_p='
            graph = graph.replace('_str', 's') # Reformat graph name nw_str -> nws
        else:  # Stochastic block model
            split_val = '_extp='

        graph_info = (graph.split(split_val)[0]  # Remove p value
                      .replace('_', ', ')
                      .replace('=', ': '))  # Reformat = -> :
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
                        choices=['External p', 'p', 'Noise-level'],
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
