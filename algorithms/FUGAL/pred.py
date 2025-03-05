#Fugal Algorithm was provided by anonymous authors.
import numpy as np
import math
import torch
from tqdm.auto import tqdm
import networkx as nx
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from algorithms.FUGAL.sinkhorn import sinkhorn,sinkhorn_epsilon_scaling,sinkhorn_knopp,sinkhorn_stabilized
from scipy import stats
from feature import Feature

def plot(graph1, graph2):
    plt.figure(figsize=(12,4))
    plt.subplot(121)

    nx.draw(graph1)
    plt.subplot(122)

    nx.draw(graph2)
    plt.savefig('x1.png')

def feature_extraction(G,features):
    """Node feature extraction.

    Parameters
    ----------

    G (nx.Graph): a networkx graph.

    Returns
    -------

    node_features (float): the Nx7 matrix of node features."""

    # necessary data structures
    node_features = np.zeros(shape=(G.number_of_nodes(), len(features)))
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    node_clustering_dict = dict(nx.clustering(G))
    egonets = {n: nx.ego_graph(G, n) for n in node_list}

    neighbor_degs = [[node_degree_dict[m] for m in egonets[n].nodes if m != n]
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    neighbor_cluster = [[node_clustering_dict[m] for m in egonets[n].nodes if m != n]
            if node_degree_dict[n] > 0
            else 0
            for n in node_list
    ]

# NETSIMILE features:
    # node degrees
    if Feature.DEG in features:
        degs = [node_degree_dict[n] for n in node_list]

        node_features[:, features.index(Feature.DEG)] = degs

    # clustering coefficient
    if Feature.CLUSTER in features:
        clusts = [node_clustering_dict[n] for n in node_list]

        node_features[:, features.index(Feature.CLUSTER)] = clusts

    # average degree of neighborhood
    if Feature.AVG_EGO_DEG in features:
        avg_neighbor_degs = [np.mean(degs) for degs in neighbor_degs]

        node_features[:, features.index(Feature.AVG_EGO_DEG)] = avg_neighbor_degs


    # average clustering coefficient of neighborhood
    if Feature.AVG_EGO_CLUSTER in features:
        neighbor_clusts = [np.mean(cluster_coeffs) for cluster_coeffs in neighbor_cluster]

        node_features[:, features.index(Feature.AVG_EGO_CLUSTER)] = neighbor_clusts

    # number of edges in the neighborhood
    if Feature.EGO_EDGES in features:
        neighbor_edges = [
            egonets[n].number_of_edges() if node_degree_dict[n] > 0 else 0
            for n in node_list
        ]

        node_features[:, features.index(Feature.EGO_EDGES)] = neighbor_edges

    # number of outgoing edges from the neighborhood
    # the sum of neighborhood degrees = 2*(internal edges) + external edges
    # node_features[:,5] = node_features[:,0] * node_features[:,2] - 2*node_features[:,4]
    if Feature.EGO_OUT_EDGES in features:
        neighbor_outgoing_edges = [
            len(
                [
                    edge
                    for edge in set.union(*[set(G.edges(j)) for j in egonets[i].nodes])
                    if not egonets[i].has_edge(*edge)
                ]
            )
            for i in node_list
        ]

        node_features[:, features.index(Feature.EGO_OUT_EDGES)] = neighbor_outgoing_edges

    # number of neighbors of neighbors (not in neighborhood)
    if Feature.EGO_NEIGHBORS in features:
        neighbors_of_neighbors = [
            len(
                set([p for m in G.neighbors(n) for p in G.neighbors(m)])
                - set(G.neighbors(n))
                - set([n])
            )
            if node_degree_dict[n] > 0
            else 0
            for n in node_list
        ]

        node_features[:, features.index(Feature.EGO_NEIGHBORS)] = neighbors_of_neighbors

# Augmented NETSIMILE FEATURES
    # sum of degrees in the neighborhood
    if 'sum_ego_deg' in features:
        sum_neighbor_degs = [np.sum(degs) for degs in neighbor_degs]

        node_features[:,features.index('sum_ego_deg')] = sum_neighbor_degs

    if 'var_ego_deg' in features:
        var_neighbor_degs = [np.var(degs) for degs in neighbor_degs]

        node_features[:, features.index('var_ego_deg')] = var_neighbor_degs

    if Feature.SUM_EGO_CLUSTER in features:
        sum_neighbor_cluster = [np.sum(cluster_coeffs) for cluster_coeffs in neighbor_cluster]

        node_features[:, features.index(Feature.SUM_EGO_CLUSTER)] = sum_neighbor_cluster

    if Feature.VAR_EGO_CLUSTER in features:
        var_neighbor_cluster = [np.var(cluster_coeffs) for cluster_coeffs in neighbor_cluster]

        node_features[:, features.index(Feature.VAR_EGO_CLUSTER)] = var_neighbor_cluster

    if 'avg_ego_edges' in features:
        avg_neighbor_edges = [
            egonets[n].number_of_edges()/egonets[n].number_of_nodes() if node_degree_dict[n] > 0 else 0
            for n in node_list
        ]

        node_features[:, features.index('avg_ego_edges')] = avg_neighbor_edges

    if 'avg_ego_out_edges' in features:
        avg_neighbor_outgoing_edges = [
            len(
                [
                    edge
                    for edge in set.union(*[set(G.edges(j)) for j in egonets[i].nodes])
                    if not egonets[i].has_edge(*edge)
                ]
            ) / node_degree_dict[i]
            for i in node_list
        ]

        node_features[:, features.index('avg_ego_out_edges')] = avg_neighbor_outgoing_edges

    if 'avg_ego_neighbors' in features:
        avg_neighbors_of_neighbors = [
            len(
                set([p for m in G.neighbors(n) for p in G.neighbors(m)])
                - set(G.neighbors(n))
                - set([n])
            ) / node_degree_dict[n]
            if node_degree_dict[n] > 0
            else 0
            for n in node_list
        ]

        node_features[:, features.index('avg_ego_neighbors')] = avg_neighbors_of_neighbors


# OUR OWN FEATURES (mode, median, min, max, range, skewness, kurtosis)
    if Feature.MODE_EGO_DEGS in features:
        # stats.mode returns the mode and the count. We extract the mode with [0].
        mode_neighbor_degs = [stats.mode(degs)[0] if len(degs) > 0 else 0 for degs in neighbor_degs]

        node_features[:, features.index(Feature.MODE_EGO_DEGS)] = mode_neighbor_degs

    if Feature.MEDIAN_EGO_DEGS in features:
        median_neighbor_degs = [np.median(degs) if len(degs) > 0 else 0 for degs in neighbor_degs]

        node_features[:, features.index(Feature.MEDIAN_EGO_DEGS)] = median_neighbor_degs

    if Feature.MIN_EGO_DEGS in features:
        min_neighbor_degs = [np.min(degs) if len(degs) > 0 else 0 for degs in neighbor_degs]

        node_features[:, features.index(Feature.MIN_EGO_DEGS)] = min_neighbor_degs

    if Feature.MAX_EGO_DEGS in features:
        max_neighbor_degs = [np.max(degs) if len(degs) > 0 else 0 for degs in neighbor_degs]

        node_features[:, features.index(Feature.MAX_EGO_DEGS)] = max_neighbor_degs

    if Feature.RANGE_EGO_DEGS in features:
        range_neighbor_degs = [np.max(degs) - np.min(degs) if len(degs) > 0 else 0 for degs in neighbor_degs]

        node_features[:, features.index(Feature.RANGE_EGO_DEGS)] = range_neighbor_degs

    if Feature.SKEWNESS_EGO_DEGS in features:
        skew_neighbor_degs = [stats.skew(degs) if len(degs) > 0 else 0 for degs in neighbor_degs]

        node_features[:, features.index(Feature.SKEWNESS_EGO_DEGS)] = skew_neighbor_degs

    if Feature.KURTOSIS_EGO_DEGS in features:
        kurtosis_neighbor_degs = [stats.kurtosis(degs) if len(degs) > 0 else 0 for degs in neighbor_degs]

        node_features[:, features.index(Feature.KURTOSIS_EGO_DEGS)] = kurtosis_neighbor_degs

    # Assortativity of egonet
    if Feature.ASSORTATIVITY_EGO in features:
        assortativity_neighbors = [nx.degree_assortativity_coefficient(egonets[n]) for n in node_list
                                   ]

        node_features[:, features.index(Feature.ASSORTATIVITY_EGO)] = assortativity_neighbors


# Centrality measures

    # Calculate centrality measures for every vertex
    if Feature.CLOSENESS_CENTRALITY in features:
        closeness_centrality = [nx.closeness_centrality(G, u=node) for node in G.nodes()]

        node_features[:, features.index(Feature.CLOSENESS_CENTRALITY)] = closeness_centrality

    if Feature.DEGREE_CENTRALITY in features:
        dc_dict = nx.degree_centrality(G)
        degree_centrality = [dc_dict[node] for node in G.nodes()]

        node_features[:, features.index(Feature.DEGREE_CENTRALITY)] = degree_centrality

    if Feature.EIGENVECTOR_CENTRALITY in features:
        ec_dict = nx.eigenvector_centrality(G, tol=0.0001, max_iter=10000)
        eigenvector_centrality = [ec_dict[node] for node in G.nodes()]

        node_features[:, features.index(Feature.EIGENVECTOR_CENTRALITY)] = eigenvector_centrality

    if Feature.PAGERANK in features:
        pr_dict = nx.pagerank(G, tol=0.0001, max_iter=10000)
        pagerank = [pr_dict[node] for node in G.nodes()]

        node_features[:, features.index(Feature.PAGERANK)] = pagerank

    if 'laplacian_centrality' in features:
        laplacian_centrality = list(nx.laplacian_centrality(G).values())

        node_features[:, features.index('laplacian_centrality')] = laplacian_centrality

    if Feature.KATZ_CENTRALITY in features:
        katz_dict = nx.pagerank(G, tol=0.0001, max_iter=10000)
        katz = [katz_dict[node] for node in G.nodes()]

        node_features[:, features.index(Feature.KATZ_CENTRALITY)] = katz

# Avg effective resistance
    if 'avg_resist_dist' in features:
        avg_resist_dists = [np.mean([float(nx.resistance_distance(G, node, other)) for other in G.nodes()]) for node in G.nodes()]

        node_features[:, features.index('avg_resist_dist')] = avg_resist_dists

# Internal vs external connectivity
    if Feature.INTERNAL_FRAC_EGO in features:
        ego_in_edges = [egonets[n].number_of_edges() for n in node_list]

        ego_in_out_edges = [
            len(
                [
                    edge
                    for edge in set.union(*[set(G.edges(j)) for j in egonets[i].nodes])
                ]
            )
            for i in node_list
        ]

        frac = [in_edges/in_out_edges if in_out_edges != 0 else 1
                for in_edges, in_out_edges in zip(ego_in_edges, ego_in_out_edges)]

        node_features[:, features.index(Feature.INTERNAL_FRAC_EGO)] = frac


# Distance measures
    # NOTE: fails if the graph is not connected!
    if 'eccentricity' in features:
        ec_dict = nx.eccentricity(G)
        eccentricity = [ec_dict[node] for node in G.nodes()]

        node_features[:, features.index('eccentricity')] = eccentricity

# Features with 2-hop neighbourhood

    two_hop_neighbor_nodes = {node: list(set.union(*[set(egonets[m].nodes) for m in egonets[node].nodes]))
                                for node in node_list}

    two_hop_egonets = {node: G.subgraph(two_hop_neighbor_nodes[node])
                       for node in node_list}

    two_hop_neighbor_degs = [[node_degree_dict[m] for m in two_hop_egonets[n].nodes if m != n]
                     if node_degree_dict[n] > 0
                     else 0
                     for n in node_list
                     ]

    two_hop_neighbor_cluster = [[node_clustering_dict[m] for m in two_hop_egonets[n].nodes if m != n]
                        if node_degree_dict[n] > 0
                        else 0
                        for n in node_list
                        ]

    # NetSimile
    # average degree of 2-hop neighborhood
    if Feature.AVG_2HOP_DEG in features:
        avg_two_neighbor_degs = [np.mean(degs) for degs in two_hop_neighbor_degs]

        node_features[:, features.index(Feature.AVG_2HOP_DEG)] = avg_two_neighbor_degs

    # average clustering coefficient of 2-hop neighborhood
    if Feature.AVG_2HOP_CLUSTER in features:
        two_neighbor_clusts = [np.mean(cluster_coeffs) for cluster_coeffs in two_hop_neighbor_cluster]

        node_features[:, features.index(Feature.AVG_2HOP_CLUSTER)] = two_neighbor_clusts

        # number of edges in the 2-hop neighborhood
    if Feature.TWOHOP_EDGES in features:
        two_neighbor_edges = [
            two_hop_egonets[n].number_of_edges() if node_degree_dict[n] > 0 else 0
            for n in node_list
        ]

        node_features[:, features.index(Feature.TWOHOP_EDGES)] = two_neighbor_edges

    # sum of degrees in the 2-hop neighborhood
    if 'sum_2hop_deg' in features:
        sum_two_neighbor_degs = [np.sum(degs) for degs in two_hop_neighbor_degs]

        node_features[:, features.index('sum_2hop_deg')] = sum_two_neighbor_degs

    if Feature.TWOHOP_NEIGHBORS in features:
        neighbors_of_2hop = [
            len(
                set([p for m in two_hop_neighbor_nodes[n] for p in G.neighbors(m)])
                - set(two_hop_neighbor_nodes[n])
                - set([n])
            )
            if node_degree_dict[n] > 0
            else 0
            for n in node_list
        ]
        node_features[:, features.index(Feature.TWOHOP_NEIGHBORS)] = neighbors_of_2hop


    # Augmented NetSimile
    if 'var_2hop_deg' in features:
        var_two_neighbor_degs = [np.var(degs) for degs in two_hop_neighbor_degs]

        node_features[:, features.index('var_2hop_deg')] = var_two_neighbor_degs

    if Feature.SUM_2HOP_CLUSTER in features:
        sum_two_neighbor_cluster = [np.sum(cluster_coeffs) for cluster_coeffs in two_hop_neighbor_cluster]

        node_features[:, features.index(Feature.SUM_2HOP_CLUSTER)] = sum_two_neighbor_cluster

    if Feature.VAR_2HOP_CLUSTER in features:
        var_two_neighbor_cluster = [np.var(cluster_coeffs) for cluster_coeffs in two_hop_neighbor_cluster]

        node_features[:, features.index(Feature.VAR_2HOP_CLUSTER)] = var_two_neighbor_cluster


    if Feature.INTERNAL_FRAC_2HOP in features:
        two_hop_in_edges = [two_hop_egonets[n].number_of_edges() for n in node_list]

        two_hop_in_out_edges = [
            len(
                [
                    edge
                    for edge in set.union(*[set(G.edges(j)) for j in two_hop_egonets[i].nodes])
                ]
            )
            for i in node_list
        ]

        frac = [in_edges / in_out_edges if in_out_edges != 0 else 1
                for in_edges, in_out_edges in zip(two_hop_in_edges, two_hop_in_out_edges)]

        node_features[:, features.index(Feature.INTERNAL_FRAC_2HOP)] = frac

    # OUR OWN FEATURES (mode, median, min, max, range, skewness, kurtosis)
    if 'mode_2hop_degs' in features:
        # stats.mode returns the mode and the count. We extract the mode with [0].
        mode_two_neighbor_degs = [stats.mode(degs)[0] for degs in two_hop_neighbor_degs]

        node_features[:, features.index('mode_2hop_degs')] = mode_two_neighbor_degs

    if Feature.MEDIAN_2HOP_DEGS in features:
        median_two_neighbor_degs = [np.median(degs) for degs in two_hop_neighbor_degs]

        node_features[:, features.index(Feature.MEDIAN_2HOP_DEGS)] = median_two_neighbor_degs

    if 'min_2hop_degs' in features:
        min_neighbor_degs = [np.min(degs) for degs in two_hop_neighbor_degs]

        node_features[:, features.index('min_2hop_degs')] = min_neighbor_degs

    if Feature.MAX_2HOP_DEGS in features:
        max_neighbor_degs = [np.max(degs) for degs in two_hop_neighbor_degs]

        node_features[:, features.index(Feature.MAX_2HOP_DEGS)] = max_neighbor_degs

    if Feature.RANGE_2HOP_DEGS in features:
        range_neighbor_degs = [np.max(degs) - np.min(degs) for degs in two_hop_neighbor_degs]

        node_features[:, features.index(Feature.RANGE_2HOP_DEGS)] = range_neighbor_degs

    if Feature.SKEWNESS_2HOP_DEGS in features:
        skew_neighbor_degs = [stats.skew(degs) for degs in two_hop_neighbor_degs]

        node_features[:, features.index(Feature.SKEWNESS_2HOP_DEGS)] = skew_neighbor_degs

    if 'kurtosis_2hop_degs' in features:
        kurtosis_neighbor_degs = [stats.kurtosis(degs) for degs in two_hop_neighbor_degs]

        node_features[:, features.index('kurtosis_2hop_degs')] = kurtosis_neighbor_degs

    # Assortativity of 2-hop neighbourhood
    if Feature.ASSORTATIVITY_2HOP in features:
        assortativity_neighbors = [nx.degree_assortativity_coefficient(two_hop_egonets[n]) for n in node_list
                                    ]

        node_features[:, features.index(Feature.ASSORTATIVITY_2HOP)] = assortativity_neighbors

    node_features = np.nan_to_num(node_features)
    return np.nan_to_num(node_features)

def feature_extractionEV(G):
    """Node feature extraction.

    Parameters
    ----------

    G (nx.Graph): a networkx graph.

    Returns
    -------

    node_features (float): the Nx7 matrix of node features."""

    # necessary data structures
    node_features = np.zeros(shape=(G.number_of_nodes(), 7))
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    node_clustering_dict = dict(nx.clustering(G))
    egonets = {n: nx.ego_graph(G, n) for n in node_list}

    # node degrees
    degs = [node_degree_dict[n] for n in node_list]

    # clustering coefficient
    clusts = [node_clustering_dict[n] for n in node_list]

    # average degree of neighborhood
    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # average clustering coefficient of neighborhood
    neighbor_clusts = [
        np.mean([node_clustering_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    1
    try:
    # Calculate eigenvector centrality for the entire graph
        centrality = nx.eigenvector_centrality(G)
    except Exception as e:
    # If an error occurs, set centrality to zeros matrix
        centrality = {node: 0.0 for node in G.nodes()}

# Extract centrality values for each node and store in a list
    EC = [centrality[node] for node in G.nodes()]

    # assembling the features
    node_features[:, 0] = degs
    node_features[:, 1] = clusts
    node_features[:, 2] = neighbor_degs
    node_features[:, 3] = neighbor_clusts
    node_features[:, 4] = EC

    node_features = np.nan_to_num(node_features)
    return np.nan_to_num(node_features)

def feature_extractionBM(G,simple):
    """Node feature extraction.

    Parameters
    ----------

    G (nx.Graph): a networkx graph.

    Returns
    -------

    node_features (float): the Nx7 matrix of node features."""

    # necessary data structures
    node_features = np.zeros(shape=(G.number_of_nodes(), 7))
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    node_clustering_dict = dict(nx.clustering(G))
    egonets = {n: nx.ego_graph(G, n) for n in node_list}

    # node degrees
    degs = [node_degree_dict[n] for n in node_list]

    # clustering coefficient
    clusts = [node_clustering_dict[n] for n in node_list]

    # average degree of neighborhood
    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # average clustering coefficient of neighborhood
    neighbor_clusts = [
        np.mean([node_clustering_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # number of edges in the neighborhood

    if simple==False:
        neighbor_edges = [
            egonets[n].number_of_edges() if node_degree_dict[n] > 0 else 0
            for n in node_list
        ]

    # number of outgoing edges from the neighborhood
    # the sum of neighborhood degrees = 2*(internal edges) + external edges
    # node_features[:,5] = node_features[:,0] * node_features[:,2] - 2*node_features[:,4]
    if simple==False:
        neighbor_outgoing_edges = [
            len(
                [
                    edge
                    for edge in set.union(*[set(G.edges(j)) for j in egonets[i].nodes])
                    if not egonets[i].has_edge(*edge)
                ]
            )
            for i in node_list
        ]   

    # number of neighbors of neighbors (not in neighborhood)
    if simple==False:
        neighbors_of_neighbors = [
            len(
                set([p for m in G.neighbors(n) for p in G.neighbors(m)])
                - set(G.neighbors(n))
                - set([n])
            )
            if node_degree_dict[n] > 0
            else 0
            for n in node_list
        ]

    # assembling the features
    node_features[:, 0] = degs
    node_features[:, 1] = clusts
    #node_features[:, 2] = neighbor_degs
    #node_features[:, 3] = neighbor_clusts
    #if (simple==False):
    #    node_features[:, 4] = neighbor_edges #create if statement
    #    node_features[:, 5] = neighbor_outgoing_edges#
    #    node_features[:, 6] = neighbors_of_neighbors#

    node_features = np.nan_to_num(node_features)
    return np.nan_to_num(node_features)
def Degree_Features(G,EFN):
    CS = []
    BS = []
    DC = []
    EC = []  # Eigenvector Centrality
    PR = []  # PageRank
    KC = []  # Katz Centrality
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    degs = [node_degree_dict[n] for n in node_list]

# Calculate centrality measures for every vertex
    for node in G.nodes():
        if (EFN==0):
            CS.append(nx.closeness_centrality(G, u=node))
        #BS.append(nx.betweenness_centrality(G)[node]) nono
        elif (EFN==1):
            DC.append(nx.degree_centrality(G)[node])
        elif (EFN==2):
            EC.append(nx.eigenvector_centrality(G,tol=0.0001,max_iter=10000)[node])
        elif (EFN==3):
            PR.append(nx.pagerank(G,tol=0.0001,max_iter=10000)[node])
        #BS.append(nx.laplacian_centrality(G)[node]) nono
        
    node_features = np.zeros(shape=(G.number_of_nodes(), 1))
    if (EFN==0):
        node_features[:, 0] = CS
    elif (EFN==1):
        node_features[:, 0] = DC
    #node_features[:, 0] = BS
    elif (EFN==2):
        node_features[:, 0] = EC
    elif (EFN==3):
        node_features[:, 0] = PR
    #elif (EFN==4):
    #    node_features[:, 0] = degs
    #print(node_features)
    node_features = np.nan_to_num(node_features)
    print(node_features)
    return np.nan_to_num(node_features)


def eucledian_dist(F1, F2, n):
    D = euclidean_distances(F1, F2)
    return D

def dist(A, B, P):
    obj = np.linalg.norm(np.dot(A, P) - np.dot(P, B))
    return obj*obj/2
'''
def convex_initTun(A, B, D, mu, niter):
    np.set_printoptions(suppress=True)
    n = len(A)
    P = torch.ones((n,n), dtype = torch.float64)
    P=P/n
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0
    K=mu*D
    decreasingV=1/niter
    DV=1.0+decreasingV
    for i in range(niter):
        DV-=decreasingV
        for it in range(1, 11):
            G=-torch.mm(torch.mm(A.T, P), B)-torch.mm(torch.mm(A, P), B.T)+ K*DV+ i*(mat_ones - 2*P)
            q = sinkhorn(ones, ones, G, reg, maxIter = 500, stopThr = 1e-3)
            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
    return P
'''
def convex_initTun(A, B, D,K, mu, niter):
    np.set_printoptions(suppress=True)
    n = len(A)
    P = torch.ones((n,n), dtype = torch.float64)
    P=P/n
    #P=D
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0
    #K=mu*D*-1
    for i in range(0,11):
        for it in range(1, 11):
            #G=  A.T@torch.sign(A @ P- P@B)- torch.sign(A@P-P@B) @ B.T+K+ i*( - 2*P)
            G=-torch.mm(torch.mm(A.T, P), B)-torch.mm(torch.mm(A, P), B.T)+ mat_ones+  i*1*( - P)
            #G= A.T@A@P+P@B.T@B+2*A@P@B+0.2**2*P-K
            q = sinkhorn(ones, ones, G, reg, maxIter = 1000, stopThr = 1e-6)
            alpha = 2.0 / float(2.0 + it)
            #alpha = 0.01
            #P = P -alpha*G
            P = P + alpha * (q - P)
        #G= A.T@A@P+P@B.T@B-2*A@P@B-0.2**2*i*P-10*K
        #q = sinkhorn(ones, ones, G, reg, maxIter = 1500, stopThr = 1e-3)
        #alpha = 2.0 / float(2.0 + it)
        #P = P + alpha * (q - P)
    return P
def convex_init(A, B, D, mu, niter):
    np.set_printoptions(suppress=True)
    n = len(A)
    P = torch.ones((n,n), dtype = torch.float64)
    P=P/n
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0
    K=mu*D
    #P=sinkhorn(ones, ones, K, reg, maxIter = 1500, stopThr = 1e-3)
    #P=torch.zeros((n,n), dtype = torch.float64)
    for i in range(niter):
        for it in range(1, 11):
            #G=  A.T@torch.sign(A @ P- P@B)- torch.sign(A@P-P@B) @ B.T+K+ i*( - 2*P)
            G=-torch.mm(torch.mm(A.T, P), B)-torch.mm(torch.mm(A, P), B.T)+ K+ i*(mat_ones - 2*P)
            q = sinkhorn(ones, ones, G, reg, maxIter = 500, stopThr = 1e-3)
            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
    return P
def convex_initQAP(A, B, niter):
    n = len(A)
    P = torch.ones((n,n), dtype = torch.float64)
    P=P/n
    ones = torch.ones(n, dtype = torch.float64)
    mat_ones = torch.ones((n, n), dtype = torch.float64)
    reg = 1.0 
    for i in range(1):
        for it in range(1, 11):
            G=-torch.mm(torch.mm(A.T, P), B)-torch.mm(torch.mm(A, P), B.T) + i*(mat_ones - 2*P)
            q = sinkhorn(ones, ones, G, reg, maxIter = 500, stopThr = 1e-3)
            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)
    return P

def convertToPermHungarian(M, n1, n2):
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    n = len(M)

    P = np.zeros((n, n))
    ans = []
    for i in range(n):
        P[row_ind[i]][col_ind[i]] = 1
        if (row_ind[i] >= n1) or (col_ind[i] >= n2):
            continue
        ans.append((row_ind[i], col_ind[i]))
    return P, ans

def convertToPermGreedy(M, n1, n2):
    n = len(M)
    indices = torch.argsort(M.flatten())
    row_done = np.zeros(n)
    col_done = np.zeros(n)

    P = np.zeros((n, n))
    ans = []
    for i in range(n*n):
        cur_row = int(indices[n*n - 1 - i]/n)
        cur_col = int(indices[n*n - 1 - i]%n)
        if (row_done[cur_row] == 0) and (col_done[cur_col] == 0):
            P[cur_row][cur_col] = 1
            row_done[cur_row] = 1
            col_done[cur_col] = 1
            if (cur_row >= n1) or (cur_col >= n2):
                continue
            ans.append((cur_row, cur_col))
    return P, ans

def convertToPerm(A, B, M, n1, n2):
    P_hung, ans_hung = convertToPermHungarian(M, n1, n2)
    P_greedy, ans_greedy = convertToPermGreedy(M, n1, n2)
    dist_hung = dist(A, B, P_hung)
    dist_greedy = dist(A, B, P_greedy)
    if dist_hung < dist_greedy:
        return P_hung, ans_hung
    else:
        return P_greedy, ans_greedy


