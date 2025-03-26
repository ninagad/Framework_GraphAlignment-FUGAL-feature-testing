import os
import pandas as pd


class DataAnalysis:

    @staticmethod
    def top_performing_feature(sources):
        current_dir = (os.path.dirname(__file__))
        runs_dir = "../Server-runs"

        dfs = []
        for source in sources:
            path = os.path.join(current_dir, runs_dir, str(source), 'res', 'acc.xlsx')

            graph_df = pd.read_excel(path, index_col=[0, 1])

            # For each feature+noise combination compute the mean over the iterations
            graph_df = graph_df.mean(axis=1)

            dfs.append(graph_df)

        # Stack the dfs from each graph
        df = pd.concat(dfs, axis=0)

        # Compute the mean over the different graphs and noise levels for each feature
        df = df.groupby(level=[0]).mean()

        print(f'{df.sort_values()=}')

        max_feature = df.idxmax()

        return max_feature

    def forward_feature_selection(self):
        sources = [333, 334, 335]  # TODO: add indices of euroroad and multimagna

        feature_1 = self.top_performing_feature(sources)

        print(feature_1)


if __name__ == "__main__":
    da = DataAnalysis()

    da.forward_feature_selection()