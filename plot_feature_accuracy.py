import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import json
import numpy as np

def main():
    dfs = []
    dir_path = None

    colormaps = ['Blues', 'Greys','Greens', 'Purples'] #OrRd']

    #runs = [23,22,28, 25,26] # bio-celegans, single features
    runs = [37, 35, 33,36] # net-science, single features
    #runs = [39, 40, 41, 42] # euroroad, single features
    #runs = [45, 46, 47, 48] # highschool, single features
    for idx in runs:
        dir_path = os.path.join(os.path.dirname(__file__), f'runs\\{idx}')
        res_path = os.path.join(dir_path, 'res\\acc.xlsx')

        # Make sure the excel file is not open in Excel! Otherwise, this fails with Errno 13 permission denied.
        df = pd.read_excel(res_path)
        dfs.append(df)

    # Opening JSON file
    f = open(os.path.join(dir_path, 'config.json'))
    # returns JSON object as a dictionary
    meta_data = json.load(f)

    mu = meta_data['algs'][0][1]['mu']
    graph = meta_data['graph_names'][0]
    iters = meta_data['iters']

    for df in dfs:
        df.rename(columns={'Unnamed: 0': 'Features', 'Unnamed: 1': 'Noise-level'}, inplace=True)

        # Fill NaN values with the previous row values
        df['Features'] = df['Features'].ffill()

        df['mean'] = df.iloc[:,2:].mean(axis=1)

    # Create plot
    plt.figure(figsize=(10, 6))

    # Loop through unique features and plot each one
    for idx, df in enumerate(dfs):
        # Define colorscale for this set of features
        features = df['Features'].unique()

        cmap = plt.get_cmap(colormaps[idx])  # Get the colormap
        colors = cmap(np.linspace(0.3, 0.9, len(features)))  # Generate shades

        for i, feature in enumerate(features):

            subset = df[df['Features'] == feature]

            label = feature.strip("[']").replace("_", " ")  # Remove [, ', ] and replace _ with whitespace.
            plt.plot(subset['Noise-level'], subset['mean'], color=colors[i], marker='o', label=label)

    # Customize plot
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Noise-level')
    plt.ylabel('Accuracy')
    #plt.title('Ablation study for FUGAL features')

    plt.suptitle('Ablation study for FUGAL features', fontsize=24, x=0.40, y=0.97)
    plt.title(label =f'$\mu$: {mu}, graph: {graph}, each point avg of {iters} runs.', fontsize=12)

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.grid(True)

    path = os.path.join(os.path.dirname(__file__), f'runs\\plots\\acc_single_features_{graph}.svg')
    plt.savefig(path)
    #plt.show()

# Using the special variable
# __name__
if __name__=="__main__":
    main()


