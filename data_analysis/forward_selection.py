import json
import os
import pandas as pd

from enums.featureEnums import FeatureExtensions


def top_performing_feature(sources):
    current_dir = (os.path.dirname(__file__))
    runs_dir = "../Server-runs"

    dfs = []

    nu = None
    mu = None
    sinkhorn_reg = None

    for idx, source in enumerate(sources):
        run_dir = os.path.join(current_dir, runs_dir, str(source))
        acc_path = os.path.join(run_dir, 'res', 'acc.xlsx')
        config_path = os.path.join(run_dir, 'config.json')

        config_dict = json.load(open(config_path))
        args = config_dict['algs'][0][1]

        if idx == 0:
            nu = args['nu']
            mu = args['mu']
            sinkhorn_reg = args['sinkhorn_reg']

        else:  # Verify that all runs have the same nu, mu and sinkhorn_reg
            current_nu = args['nu']
            current_mu = args['mu']
            current_sinkhorn_reg = args['sinkhorn_reg']

            assert current_nu == nu, f"nu must be the same in all runs. Previous: {nu}, source {idx} has mu: {current_nu}"
            assert current_mu == mu, f"mu must be the same in all runs. Previous: {mu}, source {idx} has mu: {current_mu}"
            assert current_sinkhorn_reg == sinkhorn_reg, f"sinkhorn_reg must be the same in all runs. Previous: {sinkhorn_reg}, source {idx} has sinkhorn_reg: {current_sinkhorn_reg}"

        graph_df = pd.read_excel(acc_path, index_col=[0, 1])

        # Remove Katz centrality if it is there
        try:
            graph_df = graph_df.drop('[KATZ_CENTRALITY]', level=0)
        except KeyError:
            pass

        # For each feature+noise combination compute the mean over the iterations
        graph_df = graph_df.mean(axis=1).rename('avg accuracy')

        dfs.append(graph_df)

    # Stack the dfs from each graph
    df = pd.concat(dfs, axis=0)

    # Compute the mean over the different graphs and noise levels for each feature
    df = df.groupby(level=[0]).mean()
    df = df.sort_values(ascending=False)
    df.index.name = 'Feature set'

    max_feature = df.idxmax()

    df = df.reset_index()

    # Convert feature names to labels
    FE = FeatureExtensions()
    df['Feature set'] = df['Feature set'].apply(lambda x: FE.transform_feature_str_to_label(x))

    return df, max_feature, nu, mu, sinkhorn_reg


def save_to_file(df, feature, sources, nu, mu, reg, round_no):
    latex_table = pd.Series.to_latex(df, index=False)

    # Write to file
    current_dir = (os.path.dirname(__file__))
    file_path = os.path.join(current_dir, '..', 'tables', f"feature-forward-selection-{round_no}-features.txt")
    with open(file_path, "w") as file:
        file.write(f'sources used for computation: {sources}')
        file.write('\n\n')
        file.write(f'nu: {nu} \n'
                   f'mu: {mu} \n'
                   f'sinkhorn_reg: {reg} \n \n')

        file.write(f'Best feature: {feature}')
        file.write('\n\n')
        file.write(latex_table)


def forward_feature_selection():
    sources_dict = {1: [7522, 7523, 7524, 7525],
                    #2: [11427, 11428, 11429, 11430],
                    #2: [115, 116, 117, 118],
                    2: [12384, 12385, 12386, 12387],
                    #3: [12380, 12381, 12382, 12383],
                    3: [12388, 12389, 12390, 12391],
                    #4: [398, 399, 400, 401]
                    }

    for round_no, sources in sources_dict.items():
        df, feature, nu, mu, reg = top_performing_feature(sources)
        save_to_file(df, feature, sources, nu, mu, reg, round_no)


if __name__ == "__main__":
    forward_feature_selection()
