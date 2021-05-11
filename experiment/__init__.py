from sacred import Experiment
from sacred.observers import FileStorageObserver
import logging
from algorithms import gwl, conealign, grasp2 as grasp, regal, eigenalign, NSD, isorank2 as isorank, netalign, klaus

ex = Experiment("ex")

ex.observers.append(FileStorageObserver('runs'))

# create logger
logger = logging.getLogger('e')
logger.setLevel(logging.INFO)
logger.propagate = False

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

ex.logger = logger


_GW_args = {
    'opt_dict': {
        'epochs': 1,
        'batch_size': 1000000,
        'use_cuda': False,
        'strategy': 'soft',
        # 'strategy': 'hard',
        # 'beta': 0.1,
        'beta': 1e-1,
        'outer_iteration': 400,  # M
        'inner_iteration': 1,  # N
        'sgd_iteration': 300,
        'prior': False,
        'prefix': 'results',
        'display': False
    },
    'hyperpara_dict': {
        'dimension': 90,
        # 'loss_type': 'MSE',
        'loss_type': 'L2',
        'cost_type': 'cosine',
        # 'cost_type': 'RBF',
        'ot_method': 'proximal'
    },
    # 'lr': 0.001,
    'lr': 1e-3,
    # 'gamma': 0.01,
    # 'gamma': None,
    'gamma': 0.8,
    'max_cpu': 4
}

_CONE_args = {
    'dim': 128,  # clipped by Src[0] - 1
    'window': 10,
    'negative': 1.0,
    'niter_init': 10,
    'reg_init': 1.0,
    'nepoch': 5,
    'niter_align': 10,
    'reg_align': 0.05,
    'bsz': 10,
    'lr': 1.0,
    'embsim': 'euclidean',
    'alignmethod': 'greedy',
    'numtop': 10
}

_GRASP_args = {
    'laa': 2,
    'icp': False,
    'icp_its': 3,
    'q': 100,
    'k': 20,
    'n_eig': None,  # Src.shape[0] - 1
    'lower_t': 1.0,
    'upper_t': 50.0,
    'linsteps': True,
    'base_align': True
}

_REGAL_args = {
    'attributes': None,
    'attrvals': 2,
    'dimensions': 128,  # useless
    'k': 10,            # d = klogn
    'untillayer': 2,    # k
    'alpha': 0.01,      # delta
    'gammastruc': 1.0,
    'gammaattr': 1.0,
    'numtop': 10,
    'buckets': 2
}

_LREA_args = {
    'iters': 8,
    'method': "lowrank_svd_union",
    'bmatch': 3,
    'default_params': True
}

_NSD_args = {
    'alpha': 0.8,
    'iters': 20
}

_ISO_args = {
    'alpha': 0.6,
    'tol': 1e-12,
    'maxiter': 100
}

_NET_args = {
    'a': 1,
    'b': 2,
    'gamma': 0.95,
    'dtype': 2,
    'maxiter': 100,
    'verbose': True
}

_KLAU_args = {
    'a': 1,
    'b': 1,
    'gamma': 0.4,
    'stepm': 25,
    'rtype': 2,
    'maxiter': 100,
    'verbose': True
}

_algs = [
    (gwl, _GW_args, [3], "GW"),
    (conealign, _CONE_args, [-3], "CONE"),
    (grasp, _GRASP_args, [-3], "GRASP"),
    (regal, _REGAL_args, [-3], "REGAL"),
    (eigenalign, _LREA_args, [3], "LREA"),
    (NSD, _NSD_args, [30], "NSD"),

    (isorank, _ISO_args, [3], "ISO"),
    (netalign, _NET_args, [3], "NET"),
    (klaus, _KLAU_args, [3], "KLAU")
]

_acc_names = [
    "acc",
    "S3",
    "IC",
    "S3gt",
    "mnc",
]
