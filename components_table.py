from tabulate import tabulate
import pandas as pd
import os


current_dir = (os.path.dirname(__file__))
local_runs_dir = "runs"

source_idx = 89

path = os.path.join(current_dir, local_runs_dir, str(source_idx), 'res', 'components.csv')

df = pd.read_csv(path, index_col=0)
df = df.drop(['Iteration'], axis='columns').groupby(['Noise-level']).mean()
df = df.rename(columns={'Noise-level': 'Noise level', 'Connected-components': 'Avg #connected components'})


print(tabulate(df, headers='keys', tablefmt='latex_booktabs'))

