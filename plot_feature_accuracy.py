import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import json
import numpy as np
import sys
from itertools import batched

def plot(baseline_idx, idx):
    baseline_df = None
    baseline_color = None

    dfs = []

    colormaps = ['Blues', 'Greys','Greens', 'Purples'] #OrRd']
    group_sizes = [7, 4, 7, 4]
    marker_options = ['o', '^', 's', 'x', 'D', 'P', 'd']

    # Generate colorscale
    colorscale = np.empty((0, 4), float)
    markers = []
    for group_size, colormap in zip(group_sizes, colormaps):
        cmap = plt.get_cmap(colormap)  # Get the colormap
        colors = cmap(np.linspace(0.3, 0.9, group_size))  # Generate shades
        colorscale = np.vstack((colorscale, colors))

        markers = markers + marker_options[:group_size]

    server_runs_path = os.path.join((os.path.dirname(__file__)), 'Server-runs')
    res_dir_path = os.path.join(server_runs_path, f'{idx}')

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

    if baseline_idx is not None:
        # Baseline color
        baseline_color = plt.get_cmap("Set1")(0)

        # Load baseline df
        baseline_path = os.path.join(server_runs_path, f'{baseline_idx}', 'res', 'acc.xlsx')
        baseline_df = pd.read_excel(baseline_path)

        dfs.append(baseline_df)

    for df_ in dfs:
        df_.rename(columns={'Unnamed: 0': 'Features', 'Unnamed: 1': 'Noise-level'}, inplace=True)

        # Fill NaN values with the previous row values
        df_['Features'] = df_['Features'].ffill()

        df_['mean'] = df_.iloc[:,2:].mean(axis=1)

    # Create plot
    plt.figure(figsize=(10, 6))

    # Loop through unique features and plot each one
    # Define colorscale for this set of features
    features = df['Features'].unique()

    for i, feature in enumerate(features):

        subset = df[df['Features'] == feature]

        label = str(feature).strip("[']").replace("_", " ")  # Remove [, ', ] and replace _ with whitespace.
        plt.plot(subset['Noise-level'], subset['mean'], color=colorscale[i], marker=markers[i], label=label)

    if baseline_idx is not None:
        # Draw baseline
        label = 'baseline mu=0'
        plt.plot(baseline_df['Noise-level'], baseline_df['mean'], color=baseline_color, label=label)

    # Customize plot
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Noise-level')
    plt.ylabel('Accuracy')
    #plt.title('Ablation study for FUGAL features')

    plt.suptitle('Ablation study for FUGAL parameter $\mu$', fontsize=24, x=0.40, y=0.97)
    plt.title(label =f'$\mu$: {mu}, graph: {graph}, each point avg of {iters} runs.', fontsize=12)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', title='Features')
    plt.tight_layout()
    plt.grid(True)

    path = os.path.join(os.path.dirname(__file__), 'plots', 'mu-test', f'{graph}-mu={mu}.svg')
    plt.savefig(path)
    #plt.show()


# Give list of baseline idx and main idx as argument to script
# __name__
if __name__=="__main__":
    args = list(sys.argv[1:])

    for baseline, source in batched(args, n=2):
        if baseline != 'None':
            baseline_idx = int(baseline)
        else:
            baseline_idx = None

        source_idx = int(source)

        plot(baseline_idx, source_idx)


