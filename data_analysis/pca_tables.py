import os.path
from typing import Literal

import pandas as pd
import wandb

from utils import get_git_root


def generate_pca_table(project: Literal['pca-all-features-tuning', 'pca-15-features-tuning']):
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(f"ninagad-aarhus-university/{project}")

    summary_list = []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        summary = dict(run.summary)
        summary_list.append(summary)

    df = pd.DataFrame(summary_list)

    table_df = pd.DataFrame()
    table_df['components'] = df["pca_components"]
    table_df['cum. acc.'] = df["cum. accuracy"]
    table_df['avg. acc. (\%)'] = 100 * df["avg. accuracy"]
    table_df['avg. explained var. (\%)'] = 100 * df["avg. explained var."]

    # Sort
    table_df.sort_values("cum. acc.", ascending=False, inplace=True)

    root = get_git_root()
    path = os.path.join(root, 'tables', f'{project}.txt')
    table_df.to_latex(column_format='llrr', float_format=f"{{:0.2f}}".format, buf=path, index=False, na_rep="—")


if __name__ == "__main__":
    generate_pca_table('pca-all-features-tuning')
    generate_pca_table('pca-15-features-tuning')
