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

def filter_negative_diff(runs):
    diffs = []

    fugal_graph_name = get_graph_names_from_file([runs[0]])[0]
    fugal_graph_name = strip_graph_name(fugal_graph_name)
    fugal_df = get_acc_file_as_df(runs[0])
    fugal_df = fugal_df.replace(-1, np.nan)

    pca_graph_name = get_graph_names_from_file([runs[1]])[0]
    pca_graph_name = strip_graph_name(pca_graph_name)
    pca_df = get_acc_file_as_df(runs[1])
    pca_df = pca_df.replace(-1, np.nan)

    for i in range(6):
        for j in range(5):
            if pca_df.iloc[i, j] - fugal_df.iloc[i, j] < 0:
                print("graph: ", pca_graph_name, " ",  i, ",", j, " ", pca_df.iloc[i, j] - fugal_df.iloc[i, j])
                diffs.append(pca_df.iloc[i, j] - fugal_df.iloc[i, j])

    return diffs

def diff_of_single_graph(runs):
    diffs = []

    fugal_graph_name = get_graph_names_from_file([runs[0]])[0]
    fugal_graph_name = strip_graph_name(fugal_graph_name)
    fugal_df = get_acc_file_as_df(runs[0])
    fugal_df = fugal_df.replace(-1, np.nan)
    fugal_df['avg accuracy'] = fugal_df.mean(axis=1)

    pca_graph_name = get_graph_names_from_file([runs[1]])[0]
    pca_graph_name = strip_graph_name(pca_graph_name)
    pca_df = get_acc_file_as_df(runs[1])
    pca_df = pca_df.replace(-1, np.nan)
    pca_df['avg accuracy'] = pca_df.mean(axis=1)

    print("fugal: ", fugal_df['avg accuracy'])
    print("pca_df: ", pca_df['avg accuracy'])
    print("diff 10: ", pca_df.iloc[2, 5] - fugal_df.iloc[2, 5])
    print("diff 15: ", pca_df.iloc[3, 5] - fugal_df.iloc[3, 5])
    print("diff 20: ", pca_df.iloc[4, 5] - fugal_df.iloc[4, 5])
    print("diff 25: ", pca_df.iloc[5, 5] - fugal_df.iloc[5, 5])

    for i in range(6):
        iter_diff = []
        for j in range(5):
            iter_diff.append(fugal_df.iloc[i,j]-pca_df.iloc[i,j])
        diffs.append(np.max(iter_diff))

    print("Graph: ", fugal_graph_name, pca_graph_name)
    #print("The differences are:")
    #print(diffs)
    #print("The maximum difference is: ", np.max(diffs))

    return 0


def select_plots(filter_plot):
    baselines = [17241, 17242, 17243, 17244, 17245, 17246, 16189, 16385, 16387, 16388,
                 17238, 17248, 17239, 17247, 17240, 17252, 17251, 17253, 17254, 17255,
                 17256, 17257, 17258, 17259, 17260, 17261, 17262, 17263, 17264, 712] # aves-wildbird and mouse graph is missing
    #baselines = [17247, 17245, 17243, 17241, 17239, 17242]
    fugal_sources = [17243, 17242]
    pca_sources = [22321, 22324]

    included_baselines =[]
    for (fugal_source, pca_source) in zip(fugal_sources, pca_sources):
        graph_name = filter_plot([fugal_source, pca_source])
        if graph_name is not None:
            included_baselines.append(graph_name)

    print(included_baselines)
    print("The number of graphs is: ", len(included_baselines))

if __name__ == "__main__":
    #select_plots(filter_all_acc)
    #select_plots(diff_of_single_graph)
    #select_plots(filter_negative_diff)

    baselines = [17247, 17245, 17243, 17241, 17239, 17242]
    #fugal_sources = [21209, 21476, 21485, 21490, 21574, 21612]
    pca_sources = [22319, 22320, 22321, 22322, 22323, 22324]
    diff_of_single_graph([baselines[5], pca_sources[5]])
