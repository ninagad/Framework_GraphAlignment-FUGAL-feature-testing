import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import json
import numpy as np
import sys
from itertools import batched

from feature import FeatureExtensions as FE
from plot_utils import PlotUtils as PU

def plot(baseline, source, title, outputdir):
    pu = PU()  # Instantiate Plot Utils object

    baseline_df = None
    baseline_color = None

    dfs = []

    server_runs_path = os.path.join((os.path.dirname(__file__)), 'Server-runs')
    res_dir_path = os.path.join(server_runs_path, f'{source}')

    res_path = os.path.join(res_dir_path, 'res', 'acc.xlsx')

    # Make sure the excel file is not open in Excel! Otherwise, this fails with Errno 13 permission denied.
    df = pd.read_excel(res_path)
    dfs.append(df)

    # Opening JSON file
    f = open(os.path.join(res_dir_path, 'config.json'))
    # returns JSON object as a dictionary
    meta_data = json.load(f)

    mu = meta_data['algs'][0][1]['mu']
    graph = meta_data['graph_names'][0]
    iters = meta_data['iters']

    if baseline is not None:
        # Baseline color
        baseline_color = plt.get_cmap("Set1")(0)

        # Load baseline df
        baseline_path = os.path.join(server_runs_path, f'{baseline}', 'res', 'acc.xlsx')
        baseline_df = pd.read_excel(baseline_path)

        dfs.append(baseline_df)

    for df_ in dfs:
        df_.rename(columns={'Unnamed: 0': 'Features', 'Unnamed: 1': 'Noise-level'}, inplace=True)

        # Fill NaN values with the previous row values
        df_['Features'] = df_['Features'].ffill()

        df_['mean'] = df_.iloc[:,2:].mean(axis=1)

    twohop_colors = {'avg_2hop_deg': 2,
                     'avg_2hop_cluster': 3,
                     '2hop_edges': 4,
                     '2hop_neighbors': 6,
                     'sum_2hop_cluster': 7,
                     'var_2hop_cluster': 8,
                     'assortativity_2hop': 9,
                     'internal_frac_2hop': 10,
                     'median_2hop_degs': 12,
                     'max_2hop_degs': 14,
                     'range_2hop_degs': 15,
                     'skewness_2hop_degs': 16}

    # Create plot
    plt.figure(figsize=(10, 6))

    # Loop through unique features and plot each one
    # Define colorscale for this set of features
    features = df['Features'].unique()

    for i, feature in enumerate(features):

        subset = df[df['Features'] == feature]

        # Align 2hop colors with ego colors
        #if '2hop' in str(feature).lower():
            #idx = twohop_colors[feature.strip("[']")]
            #color = colorscale[idx]
            #marker = markers[idx]
        #else:

        if ',' not in feature: # It is a single feature
            feature = FE.to_feature(feature)
            color = pu.to_color(feature)
            marker = pu.to_marker(feature)

            label = FE.to_label(feature)
        else:
            raise NotImplementedError

        plt.plot(subset['Noise-level'], subset['mean'], color=color, marker=marker, label=label)

    if baseline is not None:
        # Draw baseline
        label = 'baseline mu=0'
        plt.plot(baseline_df['Noise-level'], baseline_df['mean'], color=baseline_color, label=label)

    # Customize plot
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Noise-level')
    plt.ylabel('Accuracy')
    #plt.title('Ablation study for FUGAL features')

    #plt.suptitle('Ablation study for FUGAL parameter $\mu$', fontsize=24, x=0.40, y=0.97)
    plt.suptitle(title, fontsize=24, x=0.40, y=0.97)
    plt.title(label =f'$\mu$: {mu}, graph: {graph}, each point avg of {iters} runs.', fontsize=12)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', title='Features')
    plt.tight_layout()
    plt.grid(True)

    path = os.path.join(os.path.dirname(__file__), 'plots', outputdir, f'{graph}-mu={mu}.svg')
    plt.savefig(path)
    #plt.show()


# Give list of baseline idx and main idx as argument to script
# __name__
if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--baseline',
                        type=int,
                        help='The index of the file used for the baseline')

    parser.add_argument('--source',
                        type=int,
                        help='The index of the file used for the main plot')

    parser.add_argument('--title',
                        type=str,
                        default='Ablation study for FUGAL features$',
                        help='The title of the plot, remember to use " " ')

    parser.add_argument('--outputdir',
                        type=str,
                        default='',
                        help='The directory the plot should be saved in')

    args = parser.parse_args()

    plot(baseline=args.baseline, source=args.source, title=args.title, outputdir=args.outputdir)


    #args = list(sys.argv[1:])

    #if len(args) > 0:
    #    source = args[0]
    #    source_idx = int(source)

    #if len(args) > 1:
    #    baseline = args[1]
    #    baseline_idx = int(baseline)

    #if len(args) > 2:
    #    title = args[2]



    #for baseline, source in batched(args, n=2):
    #    if baseline != 'None':
    #        baseline_idx = int(baseline)
    #    else:
    #        baseline_idx = None



    #   plot(baseline_idx, source_idx)


