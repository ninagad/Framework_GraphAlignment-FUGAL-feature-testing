import numpy as np
from data_analysis.utils import get_acc_file_as_df, get_graph_names_from_file, strip_graph_name


def filter_large_decrease_in_acc(baseline):
    graph_name = get_graph_names_from_file([baseline])[0]
    graph_name = strip_graph_name(graph_name)

    baseline_df = get_acc_file_as_df(baseline)
    baseline_df = baseline_df.replace(-1, np.nan)

    baseline_df['avg accuracy'] = baseline_df.mean(axis=1)


    if baseline_df.iloc[0,5] - baseline_df.iloc[5,5] > 0.25:
    #if baseline_df.iloc[0, 5] - baseline_df.iloc[5, 5] < 0.1 and (baseline_df < 0.8).all().all():
        return graph_name
    else:
        return None


def filter_all_acc(baseline):
    graph_name = get_graph_names_from_file([baseline])[0]
    graph_name = strip_graph_name(graph_name)

    baseline_df = get_acc_file_as_df(baseline)
    baseline_df = baseline_df.replace(-1, np.nan)

    if (baseline_df < 0.4).all().all():
        return graph_name
    else:
        return None

def diff_of_single_noise_level(runs):
    acc = []
    graph_name = ""
    for run in runs:
        graph_name = get_graph_names_from_file([run])[0]
        graph_name = strip_graph_name(graph_name)
        df = get_acc_file_as_df(run)
        df = df.replace(-1, np.nan)
        df['avg accuracy'] = df.mean(axis=1)
        acc.append(df.iloc[0, 5])

    print(graph_name, ": ", acc)
    diff = np.max(acc) - np.min(acc)
    print("Lowest performing: ", np.argmin(acc))
    print("Difference: ", diff, "\n")

    return diff


def select_plots(filter_plot):
    #baselines = [17241, 17242, 17243, 17244, 17245, 17246, 16189, 16385, 16387, 16388,
    #             17238, 17248, 17239, 17247, 17240, 17252, 17251, 17253, 17254, 17255,
    #             17256, 17257, 17258, 17259, 17260, 17261, 17262, 17263, 17264, 712] # aves-wildbird and mouse graph is missing
    #baselines = [17247, 17245, 17243, 17241, 17239, 17242]
    fugal_sources = [21209, 21476, 21485, 21490, 21574, 21612]
    pca_sources = [22319, 22320, 22321, 22322, 22323, 22324]
    included_baselines =[]
    for (baseline, fugal_source, pca_source) in zip(fugal_sources, pca_sources):
        graph_name = filter_plot([baseline, fugal_source, pca_source])
        if graph_name is not None:
            included_baselines.append(graph_name)

    print(included_baselines)
    print("The number of graphs is: ", len(included_baselines))

if __name__ == "__main__":
    #select_plots(filter_all_acc)
    select_plots(diff_of_single_noise_level)
