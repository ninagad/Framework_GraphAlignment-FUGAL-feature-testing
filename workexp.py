# from algorithms import regal, eigenalign, conealign, netalign, NSD, klaus, gwl, grasp2 as grasp, isorank2 as isorank
import pandas as pd

import algorithms
# from experiment import ex, _CONE_args, _GRASP_args, _GW_args, _ISO_args, _KLAU_args, _LREA_args, _NET_args, _NSD_args, _REGAL_args
from experiment import ex, _algs, _acc_names
from experiment.special_settings import *
from experiment.experiments import *
from experiment.run import run_exp
from experiment.save import plotS_G, plot_G, save

from generation.generate import init1, init2, loadnx

import signal
import subprocess

import numpy as np
import scipy.sparse as sps
import networkx as nx

import sys
import os
import pickle

# Get root of the project from git and set it as current directory.
# Necessary when running workexp.py from different directories
project_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode("utf-8")
os.chdir(project_root)


@ex.config
def global_config():
    use_largest_connected_component: bool = False

    load = []
    plot = []
    verbose = False

    run = [
        0,  # gwl
        1,  # conealign
        2,  # grasp
        3,  # regal
        4,  # eigenalign
        5,  # NSD

        6,  # isorank
        # 7,      # netalign
        # 8,      # klaus
        9,  # SGWL
        10,  # Grampa
        11,  # Grasp-B
        12,  # Fugal
        13,  # FAQ -approx
        14,  # GOT
        15,  # FGOT
        16,  # Parrot
        17,  # Path
        18,  # DS++
        19,  # MDS
        # 14, #GrampaS
        # 15,#Fugal2

    ]

    accs = [
        0,  # acc
        # 1,      # EC
        # 2,      # ICS
        # 3,      # S3
        # 4,      # jacc
        # 5,      # mnc
        # 6, #frob
    ]

    algs = [_algs[i] for i in run]

    mall = False

    if mall:
        algs = [
            (alg, args, [1, 2, 3, 30, -1, -2, -3, -30, -96, 96], algname) for alg, args, _, algname in algs
        ]

    acc_names = [_acc_names[i] for i in accs]

    tmp = []


def configg():
    tmp = [
        # list(range(50)) for _ in range(6)
        # list(range(100)) for _ in range(3)
        # list(range(150)) for _ in range(2)
        list(range(50)) for _ in range(10)
    ]

    tmp = np.array(tmp)

    # tmp = tmp.T

    tmp = tmp.flatten()

    G = nx.generators.degree_seq.configuration_model(tmp.tolist(), nx.Graph)

    G.remove_edges_from(nx.selfloop_edges(G))

    return lambda x: x, (G,)


@ex.named_config
def playground():
    graph_names = [
        # "gnp",
        "barabasi",
        # "barabasi2",
        # "barabasi3",
        # "powerlaw",
        # "arenas",
        # "LFR_span",
        # "facebook",
        # "astro",
        # "yeast5"
        # "k_normal"
    ]

    # print(tmp)

    graphs = [
        # configg(),
        # (nx.newman_watts_strogatz_graph, (20, 3, 0.5)),
        # (nx.newman_watts_strogatz_graph, (100, 3, 0.0)),
        # (nx.watts_strogatz_graph, (100, 10, 0.5)),
        # (nx.gnp_random_graph, (35000, 0.0003)),
        # (nx.gnp_random_graph, (1000, 0.01)),
        # (nx.random_regular_graph, (1000, 0.01)),

        # (nx.barabasi_albert_graph, (500, 3)),
        (nx.barabasi_albert_graph, (1133, 5)),
        # (nx.barabasi_albert_graph, (2000, 3)),
        # (nx.powerlaw_cluster_graph, (100, 2, 0.3)),

        # (lambda x:x, [[
        #     "data/real world/contacts-prox-high-school-2013/contacts-prox-high-school-2013_100.txt",
        #     "data/real world/contacts-prox-high-school-2013/contacts-prox-high-school-2013_99.txt",
        #     None
        # ]])

        # (nx.relaxed_caveman_graph, (20, 5, 0.2)),

        # (nx.stochastic_block_model, (
        #     [15, 15, 25],
        #     [
        #         [0.25, 0.05, 0.02],
        #         [0.05, 0.35, 0.07],
        #         [0.02, 0.07, 0.40],
        #     ]
        # )),
        # (nx.LFR_benchmark_graph, (1133, 3.5, 1.05,
        #                           0.1, 5.0, None, None, 50, 200, 1e-07, 2000)), #2k, desc
        # (nx.LFR_benchmark_graph, (1133, 2.5, 1.05,
        #                           0.1, 4.0, None, None, 300, 800, 1e-07, 2000)), #1600, very desc
        # (nx.LFR_benchmark_graph, (1133, 2.5, 1.05,
        #                           0.2, 4.0, None, None, 3, 1100, 1e-07, 500)),  # ~2000, very steep

        # (nx.LFR_benchmark_graph, (1133, 3, 1.05,
        #                           0.2, 5, None, None, 3, 1100, 1e-07, 500)),  # almost 3k, very desc

        # (nx.LFR_benchmark_graph, (1133, 3, 1.05,
        #                           0.2, 5.5, None, None, 3, 1100, 1e-07, 500)),  # 3500, very desc (shifting)

        # (nx.LFR_benchmark_graph, (1133, 2.75, 1.2,
        #                           0.2, 7, None, None, 3, 1100, 1e-07, 5000)),  # the 5k.. very desc + 331

        # (loadnx, ('data/arenas_old/source.txt',)),
        # (loadnx, ('data/in-arenas.txt',)),
        # (loadnx, ('data/soc-facebook.txt',)),
        # (loadnx, ('data/CA-AstroPh.txt',)),

        # (lambda x: x, ('data/arenas_old/source.txt',)),
        # (lambda x: x, ('data/arenas.txt',)),
        # (lambda x: x, ('data/CA-AstroPh.txt',)),
        # (lambda x: x, ('data/facebook.txt',)),
    ]

    # no_disc = False

    iters = 5

    noises = [
        0.00,

        # 0.01,
        # # 0.02,
        # 0.03,
        # # 0.04,
        0.05,

        # 0.06,
        # 0.07,
        # 0.08,
        # 0.09,
        # 0.10,

        # 0.05,
        # 0.10,
        # 0.15,
        # 0.20,
        # 0.25,
    ]

    # noise_type = 2


def load_path(_id):
    return f"runs/{_id}"


@ex.capture
def get_graphs(load):
    if len(load) > 0:
        id = load[0]
        if load[0] <= 0:
            raise Exception('Load ids should be larger than 0')

        S_G = pickle.load(open(f"runs/{id}/_S_G.pickle", "rb"))
    else:
        S_G = init1()

    if len(load) > 1:
        id = load[1]
        G = pickle.load(open(f"runs/{id}/_G.pickle", "rb"))

        if load[1] <= 0:
            raise Exception('Load ids should be larger than 0')
    else:
        G = init2(S_G)

    return S_G, G


def save_results(path, runtimes: np.array, results: np.array, components: pd.DataFrame):
    np.save(f"{path}/_time5", runtimes)
    np.save(f"{path}/_res6", results)

    os.makedirs(f"{path}/res")
    save(runtimes, results, f"{path}/res")
    components.to_csv(f"{path}/res/components.csv")


@ex.automain
def main(_run, _log, verbose, plot, nice=0, source_graphs=None, target_graphs=None, artifact_file=None):
    _run.info['id'] = _run._id
    path = f"runs/{_run._id}"

    if (source_graphs is not None) and (target_graphs is not None):
        S_G = source_graphs
        G = target_graphs
    else:
        S_G, G = get_graphs()

    pickle.dump(S_G, open(f"{path}/_S_G.pickle", "wb"))
    pickle.dump(G, open(f"{path}/_G.pickle", "wb"))

    if not verbose:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        algorithms.GWL.dev.util.logger.disabled = True

    try:
        os.nice(nice)
    except Exception:
        pass

    if len(plot) > 0 and plot[0]:
        plotS_G(S_G)

    if len(plot) > 1 and plot[1]:
        plot_G(G)

    time5, res6, components_df = run_exp(G, path, artifact_filename=artifact_file)

    save_results(path, time5, res6, components_df)
