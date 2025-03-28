from . import ex
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import collections
import os


@ex.capture
def plotG(G, name="", end=True, circular=False):
    G = nx.Graph(G)

    plt.figure(name)

    if len(G) <= 200:
        kwargs = {}
        if circular:
            kwargs = dict(pos=nx.circular_layout(G),
                          node_color='r', edge_color='b')
        plt.subplot(211)
        nx.draw(G, **kwargs)

        plt.subplot(212)

    degree_sequence = sorted([d for n, d in G.degree()],
                             reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    plt.bar(deg, cnt, width=0.80, color="b")

    # print(degreeCount)
    plt.title(
        f"{name} Degree Histogram.\nn = {len(G)}, e = {len(G.edges)}, maxd = {deg[0]}, disc = {degreeCount[0]}")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    # fig, ax = plt.subplots()
    # ax.set_xticks([d + 0.4 for d in deg])
    # ax.set_xticklabels(deg)

    plt.show(block=end)


@ex.capture
def plotS_G(S_G, _log, graph_names):
    for i, gi in enumerate(S_G):
        for g in gi:
            try:
                _log.debug([len(x)
                            for x in {frozenset(g.nodes[v]["community"]) for v in g}])
            except Exception:
                pass
            try:
                g_cc = len(max(nx.connected_components(g), key=len))
                _log.debug([g_cc, len(g.nodes)])
            except Exception:
                pass
            try:
                # plotG(g, 'Src')
                plotG(g, graph_names[i])
            except Exception:
                pass


@ ex.capture
def plot_G(G):
    for gi in G:
        for ni in gi:
            for g in ni:
                Src, Tar, _ = g
                plotG(Src.tolist(), 'Src', False)
                plotG(Tar.tolist(), 'Tar')


@ ex.capture
def saveexls(res4, dim1, dim2, dim3, dim4, filename):

    with pd.ExcelWriter(f"{filename}.xlsx") as writer:
        for i1, res3 in enumerate(res4):
            index = pd.MultiIndex.from_product(
                [dim2, dim3], names=["", ""]
            )
            pd.DataFrame(
                res3.reshape(-1, res3.shape[-1]),
                index=index,
                columns=dim4,
            ).to_excel(writer, sheet_name=str(dim1[i1]))


@ex.capture
def plotrees(res3, dim1, dim2, dim3, filename, xlabel="Noise level", plot_type=1):
    # dim2 stores the names of evaluation metrics
    # res2 stores the values of different evaluation measures
    # dim3 stores the noise-levels
    # res3

    # One color hue for each group of features
    colormaps = ['Blues', 'Greys', 'Greens', 'Purples']
    group_sizes = [7,7,7,4]

    # Generate colorscale
    colorscale = np.empty((0,4), float)
    for group_size, colormap in zip(group_sizes, colormaps):
        cmap = plt.get_cmap(colormap)  # Get the colormap
        colors = cmap(np.linspace(0.3, 0.9, group_size))  # Generate shades
        colorscale = np.vstack((colorscale, colors))


    for i1, res2 in enumerate(res3):
        plt.figure(figsize=(10, 6))
        for i2, res1 in enumerate(res2):
            if np.all(res1 >= 0):
                label = (dim2[i2]).strip("[']").replace("_", " ")  # Remove [, ', ] and replace _ with whitespace.
                plt.plot(dim3, res1, color=colorscale[i2], label=label)

        # Adjust if running on multiple graphs!
        graph = dim1[i1]

        plt.xlabel(xlabel)
        plt.xticks(dim3)  # dim3 is the noise-levels
        #plt.title("FUGAL features ablation study")
        #plt.suptitle('Ablation study for FUGAL features', fontsize=24, x=0.43, y=0.97)
        plt.suptitle('Ablation study for FUGAL parameter $\mu$', fontsize=24, x=0.43, y=0.97)
        # TODO: Find out how to get mu dynamically!
        #plt.title(label=f'$\mu$ = 1, graph = {graph}', fontsize=16)
        plt.title(label=f'graph: {graph}', fontsize=16)

        plt.grid()
        if plot_type == 1:
            plt.ylabel("Accuracy")
            plt.ylim([-0.1, 1.1])
        else:
            plt.ylabel("Time[s]")
            # plt.yscale('log')

        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', title="Features")
        plt.tight_layout()
        #plt.legend()
        plt.savefig(
            f"{filename}_{dim1[i1]}.svg")


def squeeze(res, dims, sq):

    try:
        res = np.squeeze(res, axis=sq)
        del dims[sq]
    except Exception:
        pass

    return res, dims


def trans(res, dims, T):
    return res.transpose(*T), [dims[i] for i in T]


def save_rec(res, dims, filename, plot_type=1):
    if len(res.shape) > 4:
        split_dim = 2
        res_lst = np.split(res, indices_or_sections=len(dims[split_dim]), axis=split_dim)

        for _dim, _res in zip(dims[split_dim], res_lst):
            _res = np.squeeze(_res, axis=split_dim)
            filename = filename.replace('acc', f'{_dim}')
            save_rec(_res, dims[:split_dim] + dims[split_dim+1:], f"{filename}", plot_type)
    else:
        saveexls(res, filename=filename,
                 dim1=dims[0],
                 dim2=dims[1],
                 dim3=dims[2],
                 dim4=dims[3],
                 )

        plotrees(np.mean(res, axis=3), filename=filename, plot_type=plot_type,
                 dim1=dims[0],
                 dim2=dims[1],
                 dim3=dims[2],
                 )


@ ex.capture
def save(time5, res6, output_path, noises, iters, algs, acc_names, graph_names, mt_names=["mt"], mtsq=True, acsq=True, s_trans=None):

    T = [0, 3, 4, 5, 1, 2]

    dims = [
        graph_names,
        noises,
        list(range(1, iters+1)),
        [a[3] for a in algs],
        mt_names,
        acc_names
    ]

    time6 = np.expand_dims(time5, axis=-1)


    # (g,n,i,alg,mt,acc)
    res, dims = trans(res6, dims, T)
    time, _ = trans(time6, list(range(len(T))), T)
    # (g,alg,mt,acc,n,i)

    time_alg = time.take(indices=[0], axis=2)
    ta_dims = dims.copy()
    time_m = time.take(indices=range(1, time.shape[2]), axis=2)
    tm_dims = dims.copy()

    if acsq:
        res, dims = squeeze(res, dims, 3)
        time_alg, ta_dims = squeeze(time_alg, ta_dims, 3)
        time_m, tm_dims = squeeze(time_m, tm_dims, 3)

    if mtsq:
        res, dims = squeeze(res, dims, 2)
        time_alg, ta_dims = squeeze(time_alg, ta_dims, 2)
        time_m, tm_dims = squeeze(time_m, tm_dims, 2)

    if s_trans is not None:
        res, dims = trans(res, dims, s_trans)  # (g,alg,n,i)
        time_alg, ta_dims = trans(time_alg, ta_dims, s_trans)  # (g,alg,n,i)
        time_m, tm_dims = trans(time_m, tm_dims, s_trans)  # (g,alg,n,i)

    save_rec(res, dims, f"{output_path}/acc")
    save_rec(time_alg, ta_dims, f"{output_path}/time_alg", plot_type=2)
    save_rec(time_m, tm_dims, f"{output_path}/time_matching", plot_type=2)
