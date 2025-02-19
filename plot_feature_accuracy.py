import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import json
import numpy as np

def main():
    dfs = []
    dir_path = None
    idx = 3

    colormaps = ['Blues', 'Greys','Greens', 'Purples'] #OrRd']
    group_sizes = [7, 4, 7, 4]

    # Generate colorscale
    colorscale = np.empty((0, 4), float)
    for group_size, colormap in zip(group_sizes, colormaps):
        cmap = plt.get_cmap(colormap)  # Get the colormap
        colors = cmap(np.linspace(0.3, 0.9, group_size))  # Generate shades
        colorscale = np.vstack((colorscale, colors))

    dir_path = os.path.join((os.path.dirname(__file__)), '..', 'Server-runs', f'{idx}')
    res_path = os.path.join(dir_path, 'res\\acc.xlsx')

    # Make sure the excel file is not open in Excel! Otherwise, this fails with Errno 13 permission denied.
    df = pd.read_excel(res_path)

    # Opening JSON file
    f = open(os.path.join(dir_path, 'config.json'))
    # returns JSON object as a dictionary
    meta_data = json.load(f)

    mu = meta_data['algs'][0][1]['mu']
    graph = meta_data['graph_names'][0]
    iters = meta_data['iters']

    df.rename(columns={'Unnamed: 0': 'Features', 'Unnamed: 1': 'Noise-level'}, inplace=True)

    # Fill NaN values with the previous row values
    df['Features'] = df['Features'].ffill()

    df['mean'] = df.iloc[:,2:].mean(axis=1)

    # Create plot
    plt.figure(figsize=(10, 6))

    # Loop through unique features and plot each one
    # Define colorscale for this set of features
    features = df['Features'].unique()

    for i, feature in enumerate(features):

        subset = df[df['Features'] == feature]

        label = str(feature).strip("[']").replace("_", " ")  # Remove [, ', ] and replace _ with whitespace.
        plt.plot(subset['Noise-level'], subset['mean'], color=colorscale[i], label=label)

    # Customize plot
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Noise-level')
    plt.ylabel('Accuracy')
    #plt.title('Ablation study for FUGAL features')

    plt.suptitle('Ablation study for FUGAL features', fontsize=24, x=0.40, y=0.97)
    plt.title(label =f'$\mu$: {mu}, graph: {graph}, each point avg of {iters} runs.', fontsize=12)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', title='Features')
    plt.tight_layout()
    plt.grid(True)

    path = os.path.join((os.path.dirname(__file__)), '..', 'Server-runs', f'{idx}\\acc_{graph}.svg')
    plt.savefig(path)
    #plt.show()

# Using the special variable
# __name__
if __name__=="__main__":
    main()


