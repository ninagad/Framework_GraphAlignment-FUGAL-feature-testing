import numpy as np
from data_analysis.utils import get_acc_file_as_df, get_algo_args, get_graph_names_from_file, strip_graph_name, get_git_root

def plot_filter(baseline):
    graph_name = get_graph_names_from_file([baseline])[0]
    graph_name = strip_graph_name(graph_name)

    baseline_df = get_acc_file_as_df(baseline)
    baseline_df = baseline_df.replace(-1, np.nan)

    baseline_df['avg accuracy'] = baseline_df.mean(axis=1)
    print(baseline_df)
    print(baseline_df.iloc[0,5])
    print(baseline_df.iloc[5,5])
    if baseline_df.iloc[0,5] - baseline_df.iloc[5,5] > 0.3:
        return graph_name
    else:
        return None

def select_plots():
    baselines = [17241, 17242, 17243, 17244, 17245, 17246, 16189, 16385, 16387, 16388, 17238, 17239, 17247, 17240]
    included_baselines =[]
    for baseline in baselines:
        graph_name = plot_filter(baseline)
        if graph_name is not None:
            included_baselines.append(graph_name)

    print(included_baselines)

if __name__ == "__main__":
    select_plots()