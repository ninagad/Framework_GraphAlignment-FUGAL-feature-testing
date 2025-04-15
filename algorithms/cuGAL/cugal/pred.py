import networkx as nx
import numpy as np
import scipy
import scipy.optimize
import scipy.sparse
import scipy.sparse.csgraph
import torch
try:
    import cudf
    import cugraph
except:
    pass
from tqdm.auto import tqdm
from functools import partial

from algorithms.cuGAL.cugal.adjacency import Adjacency
from algorithms.cuGAL.cugal import sinkhorn
from algorithms.cuGAL.cugal.config import Config, HungarianMethod, SinkhornMethod
from algorithms.cuGAL.cugal.profile import Profile, Phase, SinkhornProfile, TimeStamp
from algorithms.cuGAL.cugal.feature_extraction import Features, Features_extensive
from algorithms.cuGAL.cugal.sinkhorn import SinkhornState
from enums.scalingEnums import ScalingEnums

try:
    import cuda_kernels
    has_cuda = True
except ImportError:
    has_cuda = False


def add_feature_distance(gradient: torch.Tensor, features: torch.Tensor | Features) -> torch.Tensor:
    if type(features) is Features or type(features) is Features_extensive:
        gradient = features.add_distance(gradient)
    else:
        gradient += features
    return gradient


def dense_gradient(
    A: torch.Tensor,
    B: torch.Tensor,
    P: torch.Tensor,
    features: torch.Tensor | Features,
    iteration: int,
    config: Config
) -> torch.Tensor:
    reg_scalar = 1


    if config.nu is not None:
        # scaling of QAP
        print("Entered not none nu")
        ones = torch.ones(A.shape[0], device=config.device, dtype=config.dtype)
        D = features.distance_matrix()

        qap_term = torch.trace((A @ P @ B.T @ P.T))
        lap_term = torch.trace(P.T @ D)
        reg_term = torch.trace(P.T @ (ones - P))  # Implicitly assuming lambda=1

        qap_scalar = config.nu * (1 / qap_term)
        lap_scalar = config.mu * (1 / lap_term)
        reg_scalar = 1 / reg_term
        A = A * qap_scalar
        features.source *= lap_scalar
        features.target *= lap_scalar

    gradient = -A.T @ P @ B - A @ P @ B.T
    gradient = add_feature_distance(gradient, features) + iteration*reg_scalar*(1 - 2*P)
    if has_cuda and 'cuda' in str(P.device):
        cuda_kernels.regularize(gradient, P, iteration)
    else:
        gradient += iteration - iteration * 2 * P
    return gradient

def dense_gradient_with_prints(
    A: torch.Tensor,
    B: torch.Tensor,
    P: torch.Tensor,
    features: torch.Tensor | Features,
    iteration: int,
    config: Config
) -> torch.Tensor:
    reg_scalar = 1


    if config.nu is not None:
        # scaling of QAP
        print("Entered not none nu")
        ones = torch.ones(A.shape[0], device=config.device, dtype=config.dtype)
        D = features.distance_matrix()

        qap_term = torch.trace((A @ P @ B.T @ P.T))
        lap_term = torch.trace(P.T @ D)
        reg_term = torch.trace(P.T @ (ones - P))  # Implicitly assuming lambda=1

        qap_scalar = config.nu * (1 / qap_term)
        lap_scalar = config.mu * (1 / lap_term)
        reg_scalar = 1 / reg_term
        A = A * qap_scalar
        features.source *= lap_scalar
        features.target *= lap_scalar
    D = features.distance_matrix()
    print("the first QAP1: ", (-A.T @ P @ B )[0, :10], " QAP2: ",
          (- A @ P @ B.T)[0, :10], " LAP: ", config.mu * D[0, :10])

    gradient = -A.T @ P @ B - A @ P @ B.T
    gradient = add_feature_distance(gradient, features) + iteration*reg_scalar*(1 - 2*P)
    if has_cuda and 'cuda' in str(P.device):
        cuda_kernels.regularize(gradient, P, iteration)
    else:
        gradient += iteration - iteration * 2 * P
    return gradient


def sparse_gradient(
    A, B: Adjacency,
    A_transpose: Adjacency,
    B_transpose: Adjacency,
    P: torch.Tensor,
    features: torch.Tensor | Features,
    iteration: int,
) -> torch.Tensor:


    if A is A_transpose and B is B_transpose:
        gradient = B.mul(A.mul(P).T).T
        gradient *= -2
    else:
        gradient = B_transpose.mul(A_transpose.mul(P, negate_lhs=True).T).T \
            - B.mul(A.mul(P).T).T
    gradient = add_feature_distance(gradient, features)
    if has_cuda and 'cuda' in str(P.device):
        cuda_kernels.regularize(gradient, P, iteration)
    else:
        gradient += iteration - iteration * 2 * P
    return gradient


def update_quasi_permutation(
        P: torch.Tensor, K: torch.Tensor, u: torch.Tensor, v: torch.Tensor, alpha: float, sinkhorn_method: SinkhornMethod,
) -> torch.Tensor:
    scale = sinkhorn.scale_kernel_matrix_log if \
        sinkhorn_method == SinkhornMethod.LOG else sinkhorn.scale_kernel_matrix
    q = scale(K, u, v)
    q -= P
    q *= alpha
    P += q
    return q


def find_quasi_permutation_matrix(
    A, B: nx.Graph | Adjacency,
    features: torch.Tensor | Features,
    config: Config,
    profile: Profile,
) -> torch.Tensor:
    n = A.number_of_nodes()
    sinkhorn_state = SinkhornState(n, config)

    if config.use_sparse_adjacency:
        if not type(A) is Adjacency:
            assert not nx.is_directed(
                A), "graph must be undirected to use sparse adjacency (for now)"
            A = Adjacency.from_graph(A, config.device)
        if not type(B) is Adjacency:
            assert not nx.is_directed(
                B), "graph must be undirected to use sparse adjacency (for now)"
            B = Adjacency.from_graph(B, config.device)
    else:
        A = torch.tensor(nx.to_numpy_array(
            A, nodelist=sorted(A.nodes())), dtype=config.dtype, device=config.device)
        B = torch.tensor(nx.to_numpy_array(
            B, nodelist=sorted(B.nodes())), dtype=config.dtype, device=config.device)

    if type(A) is Adjacency:
        P = torch.full([A.number_of_nodes()] * 2, fill_value=1 /
                       A.number_of_nodes(), device=config.device, dtype=config.dtype)
    else:
        P = torch.full([A.shape[0]] * 2, fill_value=1 /
                       A.shape[0], device=config.device, dtype=config.dtype)

    D = features.distance_matrix()
    ones = torch.ones(A.shape[0], device=config.device, dtype=config.dtype)
    print("before optimization: QAP: ", torch.trace((A @ P @ B.T @ P.T)), " LAP: ", torch.trace(P.T @ D), " reg: ",
          torch.trace(P.T @ (ones - P)))
    gradient_function = partial(dense_gradient, A, B)
    print("the first gradient: ", gradient_function(P, features, 0, config))
    for λ in tqdm(range(config.iter_count), desc="λ"):
        for it in tqdm(range(1, config.frank_wolfe_iter_count + 1), desc="frank-wolfe", leave=False):
            start_time = TimeStamp(config.device)
            #gradient_function = partial(sparse_gradient, A, B, A, B) \
            #    if config.use_sparse_adjacency else partial(dense_gradient, A, B)
            gradient_function = partial(dense_gradient, A, B)
            gradient = gradient_function(P, features, λ, config)
            profile.log_time(start_time, Phase.GRADIENT)

            start_time = TimeStamp(config.device)
            sinkhorn_profile = SinkhornProfile()
            K, u, v = sinkhorn_state.solve(gradient, config, sinkhorn_profile)
            profile.sinkhorn_profiles.append(sinkhorn_profile)
            profile.log_time(start_time, Phase.SINKHORN)

            alpha = 2.0 / float(2.0 + it)
            if has_cuda and 'cuda' in config.device and config.dtype == torch.float32 and config.sinkhorn_method == SinkhornMethod.LOG:
                duality_gap = cuda_kernels.update_quasi_permutation_log(
                    P, K, u, v, alpha, config.sinkhorn_regularization)
            else:
                diff = update_quasi_permutation(
                    P, K, u, v, alpha, config.sinkhorn_method)
                duality_gap = abs(
                    torch.sum(gradient * (diff / alpha)).cpu().item())

            profile.frank_wolfe_iterations += 1
            if not config.frank_wolfe_threshold is None and duality_gap < config.frank_wolfe_threshold:
                break

            if not config.use_sinkhorn_warm_start:
                sinkhorn_state = SinkhornState(n, config)

    print("the last gradient: ", gradient[0, :10])
    return P


def hungarian(quasi_permutation: torch.Tensor, config: Config, profile: Profile) -> np.array:
    profile.sparsity = torch.sum(torch.le(quasi_permutation, torch.finfo(
        quasi_permutation.dtype).eps)).item() / quasi_permutation.numel()
    start_time = TimeStamp(config.device)
    match config.hungarian_method:
        case HungarianMethod.SCIPY:
            _, column_indices = scipy.optimize.linear_sum_assignment(
                quasi_permutation.cpu(), maximize=True)
        case HungarianMethod.GREEDY:
            column_indices = list()
            mask = torch.ones(quasi_permutation.size(0), device=quasi_permutation.device,
                              dtype=quasi_permutation.dtype)
            for row in quasi_permutation:
                row *= mask
                m = row.argmax()
                mask[m] = 0
                column_indices.append(m.cpu().item())
            column_indices = np.array(column_indices)
        case HungarianMethod.DENSE:
            assert has_cuda, "doesn't have cuda"
            series = cudf.Series(1 - quasi_permutation.flatten())
            _, column_indices = cugraph.dense_hungarian(
                series, quasi_permutation.size(0), quasi_permutation.size(1))
            column_indices = column_indices.to_numpy()
        case HungarianMethod.SPARSE:
            sparse = quasi_permutation.to_sparse_csr().cpu()
            sparse = scipy.sparse.csr_matrix((
                sparse.values().numpy(),
                sparse.col_indices().numpy(),
                sparse.crow_indices().numpy(),
            ))
            _, column_indices = scipy.sparse.csgraph.min_weight_full_bipartite_matching(
                sparse, maximize=True)

    profile.log_time(start_time, Phase.HUNGARIAN)
    return column_indices


def convert_to_permutation_matrix(
    quasi_permutation: torch.Tensor,
    config: Config,
    profile: Profile,
) -> tuple[np.array, list[tuple[int, int]]]:
    n = len(quasi_permutation)
    col_ind = hungarian(quasi_permutation, config, profile)
    permutation = np.zeros((n, n))
    mapping = []
    for i in range(n):
        permutation[i][col_ind[i]] = 1
        mapping.append((i, col_ind[i]))
    return quasi_permutation.cpu().numpy(), mapping


def cugal(
    source: nx.Graph,
    target: nx.Graph,
    feature_names: list,
    scaling: ScalingEnums,
    config: Config,
    profile=Profile(),
) -> tuple[np.array, list[tuple[int, int]]]:
    """Run cuGAL algorithm.

    Returns permutation matrix and mapping from source to target.
    """

    before = TimeStamp('cpu')

    # Make sure both graphs are the same size.
    node_count = max(max(source.nodes()), max(target.nodes()))
    source.add_nodes_from(set(range(node_count)) - set(source.nodes()))
    target.add_nodes_from(set(range(node_count)) - set(target.nodes()))

    if config.use_sparse_adjacency and "cuda" in config.device:
        source, target = Adjacency.from_graph(
            source, config.device), Adjacency.from_graph(target, config.device)

    # Feature extraction.
    start_time = TimeStamp(config.device)
    #features = Features.create(source, target, config) # original cugal features
    features = Features_extensive.create(source, target, config, feature_names, scaling)

    if config.safe_mode:
        assert features.source.isfinite().all(), "source feature tensor has NaN values"
        assert features.target.isfinite().all(), "target feature tensor has NaN values"

    if not config.recompute_distance:
        features = features.distance_matrix()

    profile.log_time(start_time, Phase.FEATURE_EXTRACTION)

    # Frank-Wolfe.
    quasi_permutation = find_quasi_permutation_matrix(
        source, target, features, config, profile)
    if config.safe_mode:
        assert quasi_permutation.isfinite().all(), "quasi_permutation tensor has NaN values"

    # Round to permutation matrix.
    output = convert_to_permutation_matrix(quasi_permutation, config, profile)
    if config.safe_mode:
        assert np.isfinite(output[0]).all(
        ), "output permutation matrix has NaN values"

    profile.time = TimeStamp('cpu').elapsed_seconds(before)

    if 'cuda' in config.device:
        profile.max_memory = torch.cuda.max_memory_allocated(config.device)
        torch.cuda.reset_peak_memory_stats(config.device)

    return output
