from algorithms import regal, eigenalign, conealign, netalign, NSD, klaus, gwl, grasp2 as grasp, isorank2 as isorank
import algorithms
from experiment import ex, _CONE_args, _GRASP_args, _GW_args, _ISO_args, _KLAU_args, _LREA_args, _NET_args, _NSD_args, _REGAL_args
from experiment.special_settings import *
from experiment.experiments import *
from experiment.commands import *
from experiment.generate import init1, init2, loadnx
from experiment.run import run_exp
from experiment.save import plotS_G, plot_G, savexls, plotres

import numpy as np
import scipy.sparse as sps
import networkx as nx

import sys
import os
import pickle


@ex.config
def global_config():

    GW_args = _GW_args
    CONE_args = _CONE_args
    GRASP_args = _GRASP_args
    REGAL_args = _REGAL_args
    LREA_args = _LREA_args
    NSD_args = _NSD_args

    ISO_args = _ISO_args
    NET_args = _NET_args
    KLAU_args = _KLAU_args

    GW_mtype = 4
    CONE_mtype = -4
    GRASP_mtype = -4
    REGAL_mtype = -4
    LREA_mtype = 4
    NSD_mtype = 40

    ISO_mtype = 2
    NET_mtype = 3
    KLAU_mtype = 3

    algs = [
        (gwl, GW_args, GW_mtype),
        (conealign, CONE_args, CONE_mtype),
        (grasp, GRASP_args, GRASP_mtype),
        (regal, REGAL_args, REGAL_mtype),
        (eigenalign, LREA_args, LREA_mtype),
        (NSD, NSD_args, NSD_mtype),

        (isorank, ISO_args, ISO_mtype),
        (netalign, NET_args, NET_mtype),
        (klaus, KLAU_args, KLAU_mtype)
    ]

    alg_names = [
        "GW",
        "CONE",
        "GRASP",
        "REGAL",
        "LREA",
        "NSD",

        "ISO",
        "NET",
        "KLAU"
    ]

    acc_names = [
        "acc",
        "S3",
        "IC",
        "S3gt",
        "mnc",
    ]

    run = [
        0,      # gwl
        1,      # conealign
        2,      # grasp
        3,      # regal

        4,      # eigenalign
        5,      # NSD

        # 6,      # isorank
        # 7,      # netalign
        # 8,      # klaus
    ]

    graphs = [
        (lambda x: x, ('data/arenas/source.txt',)),
    ]

    noises = [
        0.05
    ]

    tmp = []


@ex.named_config
def playground():

    # iters = 10

    graph_names = [
        # "barabasi",
        # "powerlaw",
        "arenas",
        # "LFR_span",
        # "facebook",
    ]

    # acc_names = [
    #     5, 4, 3, 2, 1
    # ]

    # alg_names = [
    #     "gw1",
    #     "gw2",
    #     "gw3",
    #     "gw4",
    #     "gw5",
    #     "gw6",
    # ]

    graphs = [
        # (nx.newman_watts_strogatz_graph, (100, 3, 0.5)),
        # (nx.watts_strogatz_graph, (100, 10, 0.5)),
        # (nx.gnp_random_graph, (50, 0.9)),
        # (nx.barabasi_albert_graph, (100, 5)),
        # (nx.powerlaw_cluster_graph, (100, 2, 0.3)),

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
        (loadnx, ('data/arenas/source.txt',)),
        # (loadnx, ('data/facebook/source.txt',)),
        # (loadnx, ('data/CA-AstroPh/source.txt',)),

        # (lambda x: x, ('data/arenas_old/source.txt',)),
        # (lambda x: x, ('data/arenas/source.txt',)),
        # (lambda x: x, ('data/CA-AstroPh/source.txt',)),
        # (lambda x: x, ('data/facebook/source.txt',)),

        # (lambda x: x, ({'dataset': 'arenas_old',
        #                 'edges': 1, 'noise_level': 5},)),

        # (lambda x: x, ({'dataset': 'arenas',
        #                 'edges': 1, 'noise_level': 5},)),

        # (lambda x: x, ({'dataset': 'CA-AstroPh',
        #                 'edges': 1, 'noise_level': 5},)),

        # (lambda x: x, ({'dataset': 'facebook',
        #                 'edges': 1, 'noise_level': 5},)),
    ]

    # no_disc = False

    noises = [
        # 0.00,

        # 0.01,
        # 0.02,
        # 0.03,
        # 0.04,
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


@ex.automain
def main(_config, _run, _log, verbose=False, load=[], plot=[], nice=10):

    path = f"runs/{_run._id}"

    def load_path(_id):
        _id = _id if _id > 0 else int(_run._id) + _id
        return f"runs/{_id}"

    try:
        if not verbose:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            algorithms.GWL.dev.util.logger.disabled = True

        try:
            os.nice(nice)
        except Exception:
            pass

        if len(load) > 0:
            S_G = pickle.load(open(f"{load_path(load[0])}/_S_G.pickle", "rb"))
            randcheck1 = load[0]
        else:
            S_G, randcheck1 = init1()

        pickle.dump(S_G, open(f"{path}/_S_G.pickle", "wb"))
        if len(plot) > 0 and plot[0]:
            plotS_G(S_G)

        if len(load) > 1:
            G = pickle.load(open(f"{load_path(load[1])}/_G.pickle", "rb"))
            randcheck2 = load[1]
        else:
            G, randcheck2 = init2(S_G)

        pickle.dump(G, open(f"{path}/_G.pickle", "wb"))
        if len(plot) > 1 and plot[1]:
            plot_G(G)

        randcheck = (randcheck1, randcheck2)
        _log.info("randcheck: %s", randcheck)
        open(f"{path}/_randcheck.txt", "w").write(str(randcheck))

        if len(load) > 2:
            res5 = np.load(f"{load_path(load[2])}/_res5.npy")
        else:
            res5 = run_exp(G, path)

        np.save(f"{path}/_res5", res5)

        os.makedirs(f"{path}/res")
        savexls(res5, f"{path}/res")
        plotres(res5, f"{path}/res")

    except Exception:
        _log.exception("")
