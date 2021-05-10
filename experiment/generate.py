from . import ex
import numpy as np
import networkx as nx


def refill_e(edges, n, amount):
    if amount == 0:
        return edges
    # print(edges)
    # ee = np.sort(edges).tolist()
    ee = {tuple(row) for row in np.sort(edges).tolist()}
    new_e = []
    check = 0
    while len(new_e) < amount:
        _e = np.random.randint(n, size=2)
        # _ee = np.sort(_e).tolist()
        _ee = tuple(np.sort(_e).tolist())
        check += 1
        if not(_ee in ee) and _e[0] != _e[1]:
            # ee.append(_ee)
            ee.add(_ee)
            new_e.append(_e)
            check = 0
            # print(f"refill - {len(new_e)}/{amount}")
        if check % 1000 == 999:
            print(f"refill - {check + 1} times in a row fail")
    # print(new_e)
    return np.append(edges, new_e, axis=0)


@ ex.capture
def remove_e(edges, noise, no_disc=True):
    if no_disc:
        bin_count = np.bincount(edges.flatten())
        rows_to_delete = []
        for i, edge in enumerate(edges):
            if np.random.sample(1)[0] < noise:
                e, f = edge
                if bin_count[e] > 1 and bin_count[f] > 1:
                    bin_count[e] -= 1
                    bin_count[f] -= 1
                    rows_to_delete.append(i)
        edges = np.delete(edges, rows_to_delete, axis=0)
    else:
        edges = edges[np.random.sample(edges.shape[0]) >= noise]
    return edges


def load_as_nx(path):
    G_e = np.loadtxt(path, int)
    G = nx.Graph(G_e.tolist())
    return np.array(G.edges)


def loadnx(path):
    G_e = np.loadtxt(path, int)
    return nx.Graph(G_e.tolist())


@ ex.capture
def noise_types(noise_level, noise_type=1):
    return [
        {'target_noise': noise_level},
        {'target_noise': noise_level, 'refill': True},
        {'source_noise': noise_level, 'target_noise': noise_level},
    ][noise_type - 1]


def generate_graphs(G, source_noise=0.00, target_noise=0.00, refill=False):

    if isinstance(G, dict):
        dataset = G['dataset']
        edges = G['edges']
        noise_level = G['noise_level']

        source = f"data/{dataset}/source.txt"
        target = f"data/{dataset}/noise_level_{noise_level}/edges_{edges}.txt"
        grand_truth = f"data/{dataset}/noise_level_{noise_level}/gt_{edges}.txt"

        Src_e = load_as_nx(source)
        Tar_e = load_as_nx(target)
        gt_e = np.loadtxt(grand_truth, int).T

        # Src = e_to_G(Src_e)
        # Tar = e_to_G(Tar_e)

        Gt = (
            gt_e[:, gt_e[1].argsort()][0],
            gt_e[:, gt_e[0].argsort()][1]
        )

        return Src_e, Tar_e, Gt
    elif isinstance(G, str):
        Src_e = load_as_nx(G)
    elif isinstance(G, nx.Graph):
        Src_e = np.array(G.edges)
    else:
        return sps.csr_matrix([]), sps.csr_matrix([]), (np.empty(1), np.empty(1))

    n = np.amax(Src_e) + 1
    nedges = Src_e.shape[0]

    gt_e = np.array((
        np.arange(n),
        np.random.permutation(n)
    ))

    Gt = (
        gt_e[:, gt_e[1].argsort()][0],
        gt_e[:, gt_e[0].argsort()][1]
    )

    Tar_e = Gt[0][Src_e]

    Src_e = remove_e(Src_e, source_noise)
    Tar_e = remove_e(Tar_e, target_noise)

    if refill:
        Src_e = refill_e(Src_e, n, nedges - Src_e.shape[0])
        Tar_e = refill_e(Tar_e, n, nedges - Tar_e.shape[0])

    return Src_e, Tar_e,  Gt


@ ex.capture
def init1(graphs, iters):

    S_G = [
        [alg(*args) for _ in range(iters)] for alg, args in graphs
    ]

    return S_G, np.random.rand(1)[0]


@ ex.capture
def init2(S_G, noises):

    G = [
        [
            [
                generate_graphs(g, **noise_types(noise)) for g in gi
            ] for noise in noises
        ] for gi in S_G
    ]

    return G, np.random.rand(1)[0]
