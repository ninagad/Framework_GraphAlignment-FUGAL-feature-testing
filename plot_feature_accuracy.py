import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import json

def main():
    df = pd.DataFrame()
    dir_path = None

    runs = [22,23,25,26,28] # bio-celegans, single features
    #runs = [33, 35, 36, 37] # net-science, single features
    #runs = [39, 40, 41, 42] # euroroad, single features
    #runs = [45, 46, 47, 48] # highschool, single features
    for idx in runs:
        dir_path = os.path.join(os.path.dirname(__file__), f'runs\\{idx}')
        res_path = os.path.join(dir_path, 'res\\acc.xlsx')

        # Make sure the excel file is not open in Excel! Otherwise, this fails with Errno 13 permission denied.
        if idx == runs[0]:
            df = pd.read_excel(res_path)
        else:
            partial_df = pd.read_excel(res_path)
            df = pd.concat([df, partial_df], ignore_index=True)

    # Opening JSON file
    f = open(os.path.join(dir_path, 'config.json'))
    # returns JSON object as a dictionary
    meta_data = json.load(f)

    mu = meta_data['algs'][0][1]['mu']
    graph = meta_data['graph_names'][0]

    df.rename(columns={'Unnamed: 0': 'Features', 'Unnamed: 1': 'Noise-level'}, inplace=True)

    # Fill NaN values with the previous row values
    df['Features'] = df['Features'].ffill()

    df['mean'] = df.iloc[:,2:].mean(axis=1)

    # Create plot
    plt.figure(figsize=(10, 6))

    name = "tab20"
    cmap = mpl.colormaps[name]  # type matplotlib.colors.ListedColormap
    colors = cmap.colors  # type list
    #plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)
    #plt.axes.set_prop_cycle(color=colors)
    # Generate N colors from the colormap
    #N = 25  # Number of colors to cycle through
    #colors = [cmap(i / (N - 1)) for i in range(N)]  # Sample evenly spaced colors

    # Set the color cycle
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)

    # Loop through unique features and plot each one
    for feature in df['Features'].unique():
        subset = df[df['Features'] == feature]
        label = feature.strip("[']").replace("_", " ")  # Remove [, ', ] and replace _ with whitespace.
        plt.plot(subset['Noise-level'], subset['mean'], marker='o', label=label)

    # Customize plot
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Noise-level')
    plt.ylabel('Accuracy')
    #plt.title('Ablation study for FUGAL features')

    plt.suptitle('Ablation study for FUGAL features', fontsize=24, x=0.43, y=0.97)
    plt.title(label =f'$\mu$ = {mu}, graph = {graph}', fontsize=16)

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


