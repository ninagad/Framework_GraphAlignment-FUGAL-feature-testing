import sys

from . import ex
from generation import similarities_preprocess
from evaluation import matching, evaluation
import numpy as np
import networkx as nx
import scipy.sparse as sps
import os
import copy
import time
import gc
# from memory_profiler import profile
import pandas as pd

import signal
import subprocess

import wandb
import logging
import json

# Global variables used when tuning hyperparameter mu in FUGAL
artifact: wandb.Artifact | None = None
wandb_graph: str | None = None
wandb_noiselvl: float | None = None
wandb_iteration: int | None = None
wandb_mu: float | None = None
wandb_nu: float | None = None
wandb_sinkhorn_reg: float | None = None


def format_output(res):
    if isinstance(res, tuple):
        sim, cost = res
    else:
        sim = res
        cost = None

    # if sim is not None:
    #     sim = sps.csr_matrix(sim)

    # if cost is not None:
    #     cost = sps.csr_matrix(cost)

    if sps.issparse(sim):
        sim = sim.toarray()
        # res[0] = sim.toarray()

    if sps.issparse(cost):
        cost = cost.toarray()
        # res[1] = cost.toarray()

    return sim, cost


# @profile
@ex.capture
def alg_exe(alg, data, args):
    return alg.main(data=data, **args)


@ex.capture
def run_alg(_alg, _data, Gt, accs, _log, _run, mall, mon=False, pstart=5):
    # random.seed(_seed)
    # np.random.seed(_seed)

    alg, args, mts, algname = _alg

    _log.debug(f"{f' {algname} ':#^35}")

    data = copy.deepcopy(_data)

    time1 = []

    # gc.disable()
    if mon:
        output_path = f"runs/{_run._id}/mon"
        os.makedirs(output_path, exist_ok=True)

        # i = 0
        # while os.path.exists(f"{output_path}/{i}/{algname}"):
        #     i += 1
        # output_path = f"{output_path}/{i}"
        # os.makedirs(output_path, exist_ok=True)
        # output_path = f"{output_path}/{algname}"
        # os.makedirs(output_path)

        i = 0
        while os.path.exists(f"{output_path}/{algname}_{i}"):
            i += 1
        output_path = f"{output_path}/{algname}_{i}"
        os.makedirs(output_path, exist_ok=True)

        time.sleep(pstart)
        proc = subprocess.Popen(
            ['python', 'monitor.py', output_path], shell=False)
        time.sleep(2)
    start = time.time()
    res = alg_exe(alg, data, args)
    time1.append(time.time() - start)
    if mon:
        proc.send_signal(signal.SIGINT)
    # gc.enable()
    # gc.collect()

    sim, cost = format_output(res)

    try:
        _run.log_scalar(f"{algname}.sim.size", sim.size)
        _run.log_scalar(f"{algname}.sim.max", sim.max())
        _run.log_scalar(f"{algname}.sim.min", sim.min())
        _run.log_scalar(f"{algname}.sim.avg", sim.data.mean())
    except Exception:
        pass
    try:
        _run.log_scalar(f"{algname}.cost.size", cost.size)
        _run.log_scalar(f"{algname}.cost.max", cost.max())
        _run.log_scalar(f"{algname}.cost.min", cost.min())
        _run.log_scalar(f"{algname}.cost.avg", cost.data.mean())
    except Exception:
        pass

    res2 = []
    for mt in mts:
        alg = f"{algname}_{mt}"
        try:

            start = time.time()
            ma, mb = matching.getmatching(sim, cost, mt)
            elapsed = time.time() - start

            res1 = evaluation.evall(ma, mb, _data['Src'],
                                    _data['Tar'], Gt, alg=alg)
        except Exception:
            if not mall:
                _log.exception("")
            elapsed = -1
            res1 = -np.ones(len(accs))

        time1.append(elapsed)
        res2.append(res1)

    time1 = np.array(time1)
    res2 = np.array(res2)

    with np.printoptions(suppress=True, precision=4):
        _log.debug("\n%s", res2.astype(float))

    _log.debug(f"{'#':#^35}")

    return time1, res2


# @profile
@ex.capture
def preprocess(Src, Tar, gt, _run, addgt=False):
    start = time.time()
    # L = similarities_preprocess.create_L(Tar, Src)
    L2 = similarities_preprocess.create_L(Src, Tar)
    # print(L.shape)

    gt1 = gt[0]
    gt0 = np.arange(gt[0].size)

    L = sps.coo_matrix((np.ones(gt0.size).tolist(
    ), (gt0.tolist(), gt1.tolist())), shape=(gt0.size, gt0.size)).toarray()

    # n = 500
    # x = 30

    # for _ in range(x):
    #     ii = np.random.permutation(n)
    #     jj = np.random.randint(0, n, n)

    #     for i, j in zip(ii, jj):
    #         L[i, j] = 1
    #         # L[i, j] = 0.1

    # # L[1] = 1

    L = sps.csr_matrix(L, dtype=float)

    print(L.size)
    print(L2.size)
    print(np.sum(L + L2 > 1))

    if addgt:
        L = L2 + L
    else:
        L = L2

    L[L > 1] = 1

    print(L.size)

    # L, _ = regal.main({"Src": Src, "Tar": Tar}, **REGAL_args)
    # L, _ = conealign.main({"Src": Src, "Tar": Tar}, **CONE_args)
    _run.log_scalar("graph.prep.L", time.time() - start)

    start = time.time()
    # S = similarities_preprocess.create_S(Tar, Src, L)
    S = similarities_preprocess.create_S(
        sps.csr_matrix(Src), sps.csr_matrix(Tar), L)
    _run.log_scalar("graph.prep.S", time.time() - start)

    li, lj, w = sps.find(L)

    return L, S, li, lj, w


@ex.capture
def run_algs(g, algs, _log, _run, prep=False, circular=False):
    Src_e, Tar_e, Gt = g
    print("Src_e: ", Src_e.tolist()[:10], "Tar_e: ", Tar_e.tolist()[:10])
    n = Gt[0].size

    # prefix = f"{output_path}/graphs/{graph_number+1:0>2d}_{noise_level+1:0>2d}_{i+1:0>2d}"
    # prefix = f""
    # Gt_m = np.c_[np.arange(n), Gt[0]]
    # np.savetxt(f"{prefix}_Src.txt", Src_e, fmt='%d')
    # np.savetxt(f"{prefix}_Tar.txt", Tar_e, fmt='%d')
    # np.savetxt(f"{prefix}_Gt.txt", Gt_m, fmt='%d')

    src = nx.Graph(Src_e.tolist())
    src_smallest_cc = len(min(nx.connected_components(src), key=len))
    src_largest_cc = len(max(nx.connected_components(src), key=len))
    src_median_cc = np.median([len(cc) for cc in nx.connected_components(src)])

    src_disc = src_largest_cc < n

    tar = nx.Graph(Tar_e.tolist())
    tar_largest_cc = len(max(nx.connected_components(tar), key=len))
    tar_smallest_cc = len(min(nx.connected_components(tar), key=len))
    tar_median_cc = np.median([len(cc) for cc in nx.connected_components(tar)])

    tar_disc = tar_largest_cc < n

    if (src_disc):
        _log.warning("Disc. Source: %s < %s", src_largest_cc, n)
        _log.warning("Smallest connected component in Source has %s nodes", src_smallest_cc)
        _log.warning("The median of the size of the connected components in Source are %s nodes", src_median_cc)

    _run.log_scalar("graph.Source.disc", src_disc)
    _run.log_scalar("graph.Source.n", n)
    _run.log_scalar("graph.Source.e", Src_e.shape[0])

    if (tar_disc):
        _log.warning("Disc. Target: %s < %s", tar_largest_cc, n)
        _log.warning("Smallest connected component in Target has %s nodes", tar_smallest_cc)
        _log.warning("The median of the size of the connected components in Target are %s nodes", tar_median_cc)

    _run.log_scalar("graph.Target.disc", tar_disc)
    _run.log_scalar("graph.Target.n", n)
    _run.log_scalar("graph.Target.e", Tar_e.shape[0])

    Src = e_to_G(Src_e, n)
    Tar = e_to_G(Tar_e, n)

    if prep:
        L, S, li, lj, w = preprocess(Src, Tar, Gt)
    else:
        L = S = sps.eye(1)
        li = lj = w = np.empty(1)

    data = {
        'Src': Src,
        'Tar': Tar,
        'L': L,
        'S': S,
        'li': li,
        'lj': lj,
        'w': w
    }

    time2 = []
    res3 = []

    for alg in algs:
        time1, res2 = run_alg(alg, data, Gt)
        time2.append(time1)
        res3.append(res2)

        if wandb.run is not None:
            args = alg[1]
            global wandb_nu
            wandb_nu = args['nu']
            global wandb_mu
            wandb_mu = args['mu']
            global wandb_sinkhorn_reg
            wandb_sinkhorn_reg = args['sinkhorn_reg']

            scaling = args['scaling']
            pca = args['pca']

            features = args['features']
            acc = res2.item()
            if acc == -1:
                # Log run as failed
                logging.error(f'numeric error occurred')
                # Log avg. accuracy as 0 to steer optimization away from this value of mu
                wandb.run.log({'nu': wandb_nu,
                               'mu': wandb_mu,
                               'sinkhorn_reg': wandb_sinkhorn_reg,
                               'Avg. accuracy': 0})
                wandb.finish(exit_code=1)
                # Terminate script immediately so the next value of mu is used
                sys.exit(1)

            summary_dict = {'nu': wandb_nu,
                            'mu': wandb_mu,
                            'sinkhorn_reg': wandb_sinkhorn_reg,
                            'accuracy': acc,
                            'features': [feature.name for feature in features],
                            'graph': wandb_graph,
                            'noise-level': wandb_noiselvl,
                            'iteration': wandb_iteration,
                            'scaling': scaling.name,
                            'pca': pca.name,
                            }

            filename = f'nu={wandb_nu}-mu={wandb_mu}-sinkhorn_reg={wandb_sinkhorn_reg}.json'
            # Step 1: Load JSON data from artifact file
            with open(filename, "r") as f:
                data = json.load(f)

            # Step 2: Append to the results list
            data.append(summary_dict)

            # Step 3: Write it back to the artifact file
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)

    return np.array(time2), np.array(res3)


def e_to_G(e, n):
    # n = np.amax(e) + 1
    nedges = e.shape[0]
    # G = sps.csr_matrix((np.ones(nedges), e.T), shape=(n, n), dtype=int)
    G = sps.csr_matrix((np.ones(nedges), e.T), shape=(n, n), dtype=np.int8)
    G += G.T
    G.data = G.data.clip(0, 1)
    # return G
    return G.toarray()


@ex.capture
def run_exp(G, output_path, noises, _log, graph_names):
    time2 = time3 = time4 = None
    time5 = []
    res3 = res4 = res5 = None
    res6 = []

    no_connected_components = []

    try:
        # os.mkdir(f'{output_path}/graphs')

        for graph_number, g_n in enumerate(G):

            _log.info("Graph:(%s/%s)", graph_number + 1, len(G))

            if wandb.run is not None:
                global wandb_graph
                wandb_graph = graph_names[graph_number]

            time4 = []
            res5 = []
            for noise_level, g_it in enumerate(g_n):

                _log.info("Noise_level:(%s/%s)", noise_level + 1, len(g_n))

                if wandb.run is not None:
                    global wandb_noiselvl
                    wandb_noiselvl = noises[noise_level]

                time3 = []
                res4 = []
                for i, g in enumerate(g_it):
                    _log.info("iteration:(%s/%s)", i + 1, len(g_it))

                    if wandb.run is not None:
                        global wandb_iteration
                        wandb_iteration = i + 1

                    time2, res3 = run_algs(g)

                    target_e = g[1]
                    target_g = nx.Graph(target_e.tolist())
                    connected_components = nx.number_connected_components(target_g)
                    no_connected_components.append([noises[noise_level], i, connected_components])

                    with np.printoptions(suppress=True, precision=4):
                        _log.info("\n%s", res3.astype(float))

                    time3.append(time2)
                    res4.append(res3)

                time3 = np.array(time3)
                res4 = np.array(res4)
                with np.printoptions(suppress=True, precision=4):
                    _log.debug("\n%s", res4.astype(float))
                time4.append(time3)
                res5.append(res4)

            components_df = pd.DataFrame(no_connected_components,
                                         columns=['Noise-level', 'Iteration', 'Connected-components'])

            time4 = np.array(time4)
            res5 = np.array(res5)
            time5.append(time4)
            res6.append(res5)
    except:
        np.save(f"{output_path}/_time2", np.array(time2))
        np.save(f"{output_path}/_time3", np.array(time3))
        np.save(f"{output_path}/_time4", np.array(time4))
        np.save(f"{output_path}/_time5", np.array(time5))
        np.save(f"{output_path}/_res3", np.array(res3))
        np.save(f"{output_path}/_res4", np.array(res4))
        np.save(f"{output_path}/_res5", np.array(res5))
        np.save(f"{output_path}/_res6", np.array(res6))
        # _log.exception("")
        raise

    # Log avg accuracy for each graph
    if wandb.run is not None:
        if len(graph_names) != 1:
            raise ValueError(f'Run graphs one at a time for wandb tuning. Graphs: {graph_names}')

        mean_acc = np.mean(res6)
        wandb.run.log({'nu': wandb_nu,
                       'mu': wandb_mu,
                       'sinkhorn_reg': wandb_sinkhorn_reg,
                       f'{graph_names[0]} avg. acc': mean_acc})

    return np.array(time5), np.array(res6), components_df  # (g,n,i,alg,mt,acc)
