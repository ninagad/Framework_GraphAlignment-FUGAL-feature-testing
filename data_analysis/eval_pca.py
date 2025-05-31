import os.path
from typing import Literal

import pandas as pd
import wandb
from matplotlib import pyplot as plt

from data_analysis.utils import get_git_root


def get_pca_df(project: Literal['pca-all-features-tuning', 'pca-15-features-tuning']):
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(f"ninagad-aarhus-university/{project}")

    summary_list = []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        summary = dict(run.summary)
        summary_list.append(summary)

    df_summary = pd.DataFrame(summary_list)

    df = pd.DataFrame()
    df['components'] = df_summary["pca_components"]
    df['avg. acc. (\%)'] = 100 * df_summary["avg. accuracy"]
    df['avg. explained var. (\%)'] = 100 * df_summary["avg. explained var."]

    return df


def generate_pca_table(project: str, df: pd.DataFrame):
    # Sort
    df.sort_values("components", inplace=True)

    root = get_git_root()
    path = os.path.join(root, 'tables', f'{project}.txt')
    df.to_latex(column_format='llr', float_format=f"{{:0.2f}}".format, buf=path, index=False, na_rep="—")


def generate_pca_plot(title: str, traces: list[pd.DataFrame], labels: list[str]):
    colors = [plt.get_cmap('Blues')(0.7), plt.get_cmap('Greens')(0.7)]
    markers = ['o', 's']

    for trace, label, color, marker in zip(traces, labels, colors, markers):
        trace.sort_values('components', inplace=True)

        plt.plot(trace['components'], trace['avg. acc. (\%)'], label=label, color=color, marker=marker, markersize=4)

    plt.xlabel('Principal components')
    plt.ylabel('Avg. accuracy (%)')
    plt.title(title, fontsize=18)
    plt.legend()
    plt.grid(True)

    root = get_git_root()
    path = os.path.join(root, 'plots', 'FUGAL-evaluation', 'PCA', f'{title}.pdf')
    plt.savefig(path)


if __name__ == "__main__":
    df_30 = get_pca_df('pca-all-features-tuning')
    generate_pca_table('pca-all-features-tuning', df_30)

    df_15 = get_pca_df('pca-15-features-tuning')
    generate_pca_table('pca-15-features-tuning', df_15)
    generate_pca_plot('FUGAL with PCA', [df_15, df_30], ['15 features', '30 features'])
