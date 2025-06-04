import numpy as np
from numpy import inf, nan
import scipy.sparse as sps
import scipy
import networkx as nx
from algorithms.FUGAL.pred import feature_extraction, eucledian_dist
from algorithms.FUGAL.Fugal import apply_pca, apply_scaling
from enums.scalingEnums import ScalingEnums




# from algorithms import bipartiteMatching
# from algorithms.NSD.NSD import fast2, findnz1
# from data import similarities_preprocess, ReadFile
# from evaluation import evaluation
# from evaluation.evaluation import check_with_identity
# from evaluation.evaluation_design import remove_edges_directed
# from experiment.similarities_preprocess import create_L
from math import floor, log2


def create_L(A, B, lalpha=1, mind=None, weighted=True):
    n = A.shape[0]
    m = B.shape[0]

    if lalpha is None:
        return sps.csr_matrix(np.ones((n, m)))

    a = A.sum(1)
    b = B.sum(1)
    # print(a)
    # print(b)

    # a_p = [(i, m[0,0]) for i, m in enumerate(a)]
    a_p = list(enumerate(a))
    a_p.sort(key=lambda x: x[1])

    # b_p = [(i, m[0,0]) for i, m in enumerate(b)]
    b_p = list(enumerate(b))
    b_p.sort(key=lambda x: x[1])

    ab_m = [0] * n
    s = 0
    e = floor(lalpha * log2(m))
    for ap in a_p:
        while(e < m and
              abs(b_p[e][1] - ap[1]) <= abs(b_p[s][1] - ap[1])
              ):
            e += 1
            s += 1
        ab_m[ap[0]] = [bp[0] for bp in b_p[s:e]]

    # print(ab_m)

    li = []
    lj = []
    lw = []
    for i, bj in enumerate(ab_m):
        for j in bj:
            # d = 1 - abs(a[i]-b[j]) / a[i]
            d = 1 - abs(a[i]-b[j]) / max(a[i], b[j])
            if mind is None:
                if d > 0:
                    li.append(i)
                    lj.append(j)
                    lw.append(d)
            else:
                li.append(i)
                lj.append(j)
                lw.append(mind if d <= 0 else d)
                # lw.append(0.0 if d <= 0 else d)
                # lw.append(d)

                # print(len(li))
                # print(len(lj))
                # print(len(lj))

    return sps.csr_matrix((lw, (li, lj)), shape=(n, m))

# def main(A, B, L=None, alpha=0.5, tol=1e-12, maxiter=1, verbose=True):


def main(data, features, alpha=0.5, tol=1e-12, maxiter=1, verbose=True, lalpha=10000, weighted=True, pca_components=None, scaling=ScalingEnums.NO_SCALING):
    print("Isorank")
    dtype = np.float32
    Src = data['Src']
    Tar = data['Tar']
    L = data['L']
    Src1 = nx.from_numpy_array(Src)
    Tar1 = nx.from_numpy_array(Tar)
    n1 = Tar.shape[0]
    n2 = Src.shape[0]
    F1 = feature_extraction(Src1, features)
    F2 = feature_extraction(Tar1, features)
    nr_of_features = len(features)

    if pca_components is not None:
        F1, F2, explained_var = apply_pca(F1, F2, pca_components)
        nr_of_features = pca_components
    else:
        F1, F2 = apply_scaling(F1, F2, scaling)

    Sim = np.ones((n1,n2))
    D = np.ones((n1, n2, nr_of_features))

    for i in range(n1):
        for j in range(n2):
            D[i,j] = np.absolute(F1[i,:] - F2[j,:])

    max = np.max(D, axis=(0,1))

    min = np.min([np.min(F1),np.min(F2)])
    if min < 0:
        F1 -= min
        F2 -= min
        raise Exception

    for i in range(n1):
        for j in range(n2):
            if scaling == ScalingEnums.NO_SCALING:
                max = np.max(np.vstack((F1[i, :], F2[j, :])), axis=0)
            diff = np.absolute(F1[i, :] - F2[j, :])
            for k in np.argwhere(max == 0):
                max[k] = 1
                diff[k] = 0
            #Sim[i,j] = eucledian_dist(F1[i,:], F2[j,:], 1) / np.max(np.sum(F1[i,:],F2[j,:]))
            # Sim[i,j] = np.sum(np.absolute(F1[i,:] - F2[j,:]) / np.max(np.vstack((F1[i,:],F2[j,:])), axis=0))
            # Sim[i,j] = 1 - (np.sum(np.absolute(F1[i,:] - F2[j,:])) / np.max([np.sum(F1[i,:]),np.sum(F2[j,:])]))
            #Sim[i, j] = np.sum(np.ones(nr_of_features) - (diff / max)) / nr_of_features
            Sim[i, j] = 1 - (np.sum(diff / max) / nr_of_features)
            # Sim[i,j] = 1- (np.sum(np.absolute(F1[i, :] - F2[j, :]) / np.max(np.vstack((F1[i, :], F2[j, :])), axis=0)) / nr_of_features)
            #Sim[i, j] = np.sum(np.ones(nr_of_features) - (np.absolute(F1[i, :] - F2[j, :]) / np.max(np.vstack((F1[i, :], F2[j, :])), axis=0)))
            #Sim[i, j] = np.sqrt(np.sum(np.ones(nr_of_features) - ((diff ** 2) / (max ** 2)))) / nr_of_features

    #D = eucledian_dist(F1, F2, Src.shape[0])

    if features is None:
        L = create_L(Src, Tar, lalpha=lalpha, weighted=weighted).toarray().astype(dtype)
    else:
        L = Sim

    #L = np.max(D) - D

    # normalize the adjacency matrices
    d1 = 1 / Tar.sum(axis=1)
    d2 = 1 / Src.sum(axis=1)

    d1[d1 == inf] = 0
    d2[d2 == inf] = 0
    d1 = d1.reshape(-1, 1)
    d2 = d2.reshape(-1, 1)

    W1 = d1*Tar
    W2 = d2*Src
    W2aT = (alpha*W2.T).astype(dtype)
    K = ((1-alpha) * L).astype(dtype)
    W1 = W1.astype(dtype)

    S = np.ones((n2, n1), dtype=dtype) / (n1 * n2)  # Map target to source
    # IsoRank Algorithm in matrix form
    for it in range(1, maxiter + 1):
        prev = S.flatten()
        if alpha is None:
            S = W2.T.dot(S).dot(W1)
        else:
            S = W2aT.dot(S).dot(W1) + K
        delta = np.linalg.norm(S.flatten()-prev, 2)
        #if verbose:
        #    print("Iteration: ", it, " with delta = ", delta)
        if delta < tol:
            break

    return S