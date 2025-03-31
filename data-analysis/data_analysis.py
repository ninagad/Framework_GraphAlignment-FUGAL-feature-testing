import os
import pandas as pd
import sys

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from enums.featureEnums import FeatureExtensions


class DataAnalysis:
    def transform_feature_str_to_label(self, feature: str):
        fe = FeatureExtensions()

        if ',' not in feature:
            return fe.to_label(fe.to_feature(feature))
        else:
            features_in_combination = feature.replace(' ', '').split(',')
            features = [fe.to_feature(name) for name in features_in_combination]
            return fe.to_labels(features)


    def top_performing_feature(self, sources):
        current_dir = (os.path.dirname(__file__))
        runs_dir = "../Server-runs"

        dfs = []
        for source in sources:
            path = os.path.join(current_dir, runs_dir, str(source), 'res', 'acc.xlsx')

            graph_df = pd.read_excel(path, index_col=[0, 1])

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
        df.index.name = 'Feature'

        max_feature = df.idxmax()

        df = df.reset_index()

        # Convert feature names to labels
        df['Feature'] = df['Feature'].apply(lambda x: self.transform_feature_str_to_label(x))

        return df, max_feature

    @staticmethod
    def save_to_file(df, feature, sources, round):
        latex_table = pd.Series.to_latex(df, index=False)

        # Write to file
        file_path = os.path.join('..', 'tables', f"feature-forward-selection-{round}-features.txt")
        with open(file_path, "w") as file:
            file.write(f'sources used for computation: {sources}')
            file.write('\n\n')
            file.write(f'Best feature: {feature}')
            file.write('\n\n')
            file.write(latex_table)

    def forward_feature_selection(self):

        # Collective robust scaling
        sources_1 = [369, 370, 371, 372]  # Bio-celegans, netscience, euroroad, voles
        df_1, feature_1 = self.top_performing_feature(sources_1)
        self.save_to_file(df_1, feature_1, sources_1, 1)

        # Second feature
        sources_2 = [389, 390, 392, 391]
        df_2, feature_2 = self.top_performing_feature(sources_2)
        self.save_to_file(df_2, feature_2, sources_2, 2)


if __name__ == "__main__":
    da = DataAnalysis()

    da.forward_feature_selection()
