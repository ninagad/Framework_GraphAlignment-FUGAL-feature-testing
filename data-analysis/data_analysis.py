import os
import pandas as pd
import sys

# Add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from enums.featureEnums import FeatureExtensions


class DataAnalysis:

    @staticmethod
    def top_performing_feature(sources):
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
        fe = FeatureExtensions()
        df['Feature'] = df['Feature'].apply(lambda x: fe.to_label(fe.to_feature(x)))

        print(pd.Series.to_latex(df, index=False))

        return max_feature

    def forward_feature_selection(self):
        # TODO: correct indices to new runs for correct scaling
        # Single feature for distance normalization
        sources = [330, 331, 346, 329]  # bio-celegans, netscience, euroroad, voles.

        feature_1 = self.top_performing_feature(sources)

        print(feature_1)


if __name__ == "__main__":
    da = DataAnalysis()

    da.forward_feature_selection()
