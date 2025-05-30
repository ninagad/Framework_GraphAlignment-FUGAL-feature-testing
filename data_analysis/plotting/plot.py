import os
import json
import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from enums.featureEnums import FeatureExtensions as FE
from .plot_utils import PlotUtils as PU
from data_analysis.utils import get_git_root


def load_data(source: int, xaxis: str, yaxis: str) -> (pd.DataFrame, int, str, int):
    """
    Loads data from excel file

    Args:
        source: index of directory to load from

    Returns:
        dataframe, mu, graph name and #iterations.

    """
    root_path = get_git_root()
    server_runs_path = os.path.join(root_path, 'Server-runs')
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

    # TODO: refactor to use transform_feature_str_to_label function in FeatureExtensions
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

        try:
            color = pu.to_colors(features)
        except KeyError:
            color = None
        try:
            marker = pu.to_markers(features)
        except KeyError:
            marker = None

        label = FE.to_labels(features)

    return color, marker, label


def plot(xaxis: str, yaxis: str, baseline: int, source: int, title: str, outputdir: str):
    """

    Args:
        plottype:
        baseline:
        sources:
        title:
        outputdir:

    Returns:

    """
    df, mu, graphs, iters = load_data(source, xaxis, yaxis)
    df = transform_df(df)

    xname = PU().to_column_name(xaxis)

    # Create plot
    plt.figure(figsize=(10, 6))

    # Loop through unique features and plot each one
    # Define colorscale for this set of features
    features = df['Features'].unique()

    for i, feature in enumerate(features):

        subset = df[df['Features'] == feature]

        color, marker, label = get_color_marker_label(feature)

        if (label is None) or (label is FE.to_label(FE.to_feature("KATZ_CENTRALITY"))):
            continue

        # Don't plot feature if it is not an ENUM feature
        #if label is None:
            #continue

        # If the label is more than 22 characters, split it into two lines
        max_label_len = 22
        if len(label) > max_label_len:
            label = ',\n'.join(label.rsplit(', ', 1))

        plt.plot(subset[xname], subset['mean'], color=color, marker=marker, label=label)

    # Draw baseline
    if baseline is not None:
        # Load baseline df
        baseline_df, b_mu, b_graphs, b_iters = load_data(baseline, xaxis, yaxis)
        baseline_df = transform_df(baseline_df)

        # If the Feature is NaN -> rename to dummy name 1.
        if baseline_df['Features'].isnull().all():
            baseline_df['Features'] = 1

        # Check that graph name and #iterations match source
        if (graphs[0] != b_graphs[0]): #or (iters != b_iters):
            raise Exception(
                f"The baseline does not have the same meta data graphs: {b_graphs}, iterations: {b_iters} as the main source graphs:{graphs}, iterations: {iters} :(")

        # Baseline trace color (red)
        baseline_color = plt.get_cmap("Set1")(0)

        # If there are multiple traces in the baseline file -> use the first one only
        baseline_df = baseline_df[(baseline_df['Features'] == baseline_df['Features'].iloc[0])]

        label = 'baseline (no features)'

        plt.plot(baseline_df[xname], baseline_df['mean'], color=baseline_color, label=label)

    # Layout plot
    if yaxis == 'acc':
        plt.ylim(-0.1, 1.1)
        plt.ylabel('Accuracy')

    elif yaxis == 'frob':
        plt.ylabel('Frobenius norm')
    else:
        raise NotImplementedError

    plt.xlabel(xaxis)

    plt.suptitle(title, fontsize=24, x=0.40, y=0.97)

    graph = graphs[0]
    # Format graph info in subtitle
    if xaxis == 'Noise-level':
        graph_info = graph
    else:
        if xaxis == 'p':  # NWS graph
            remove_val = '_p='
            graph = graph.replace('_str', 's') # Reformat graph name nw_str -> nws

        elif xaxis == 'External p':  # Stochastic block model
            remove_val = '_extp='

        elif xaxis == 'k':
            remove_val = '_k='
            graph = graph.replace('_str', 's')  # Reformat graph name nw_str -> nws

        elif xaxis == 'n':
            remove_val = '_n='
            graph = graph.replace('_str', 's')  # Reformat graph name nw_str -> nws

            # Reformat if k is variable
            k_0 = graphs[0].split('_k=')[1].split('_')[0]
            k_1 = graphs[1].split('_k=')[1].split('_')[0]

            if k_0 != k_1:
                n_0 = graphs[0].split('_n=')[1].split('_')[0]
                k = float(n_0) / float(k_0)
                graph = graph.replace('_k='+str(k_0), '_k='+'ndiv'+str(int(k)))

        else:
            remove_val = None

        info_split = graph.split(remove_val)
        graph = info_split[0] + ('_' + info_split[1].split('_', 1)[1] if '_' in info_split[1] else '')  # Remove variable value

        graph_info = (graph
                      .replace('_', ', ')
                      .replace('=', ': ')  # Reformat = -> :
                      .replace('div', '/'))
        graph_info += ', noise-level: ' + str(df.at[0, 'Noise-level'])  #  Add noise-level

    if mu is not None:  # It is FUGAL
        plt.title(label=f'$\mu$: {mu}, graph: {graph_info}', fontsize=12)
    else:  # Other algorithms than FUGAL
        plt.title(label=f'graph: {graph_info}', fontsize=12)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', title='Features')
    plt.tight_layout()
    plt.grid(True)

    # Save plot
    root_path = get_git_root()
    if mu is not None:
        path = os.path.join(root_path, 'plots', outputdir, f'{graph}-mu={mu}-{yaxis}-run={source}.pdf')
    else:  # For other algorithms than FUGAL
        path = os.path.join(root_path, 'plots', outputdir, f'{title}-{graph}-{yaxis}-run={source}.pdf')

    plt.savefig(path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--xaxis',
                        choices=['External p', 'p', 'k', 'Noise-level', 'n'],
                        default='Noise-level')

    parser.add_argument('--yaxis',
                        choices=['acc', 'frob'],
                        default='acc')

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

    xaxis = args.xaxis
    yaxis = args.yaxis

    baseline = args.baseline
    source = args.source
    title = args.title
    outputdir = args.outputdir

    plot(xaxis=xaxis, yaxis=yaxis, baseline=baseline, source=source, title=title, outputdir=outputdir)
