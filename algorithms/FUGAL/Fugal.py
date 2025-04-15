#Fugal Algorithm was provided by anonymous authors.
import numpy as np
from numpy import inf, nan
import scipy.sparse as sps
import scipy as sci
from math import floor, log2
import math
import torch
import networkx as nx
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA

from algorithms.FUGAL.pred import feature_extraction,eucledian_dist,convex_init,Degree_Features
from enums.pcaEnums import PCAEnums
from enums.scalingEnums import ScalingEnums

def are_matrices_equal(matrix1, matrix2):
    # Check if dimensions are the same
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        return False
    
    # Check element-wise equality
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            if matrix1[i][j] != matrix2[i][j]:
                return False

    # If no inequality is found, matrices are equal
    return True


def main(data, iter, sinkhorn_reg: float, nu: float, mu, features: list, scaling: ScalingEnums, pca: PCAEnums, EFN=5):
    print("Fugal")
    torch.set_num_threads(40)
    dtype = np.float64
    Src = data['Src']
    Tar = data['Tar']
    for i in range(Src.shape[0]):
        row_sum1 = np.sum(Src[i, :])

    # If the sum of the row is zero, add a self-loop
        if row_sum1 == 0:
            Src[i, i] = 1
    for i in range(Tar.shape[0]):
        row_sum = np.sum(Tar[i, :])

    # If the sum of the row is zero, add a self-loop
        if row_sum == 0:
            Tar[i, i] = 1
    n1 = Tar.shape[0]
    n2 = Src.shape[0]
    n = max(n1, n2)
    Src1=nx.from_numpy_array(Src)
    Tar1=nx.from_numpy_array(Tar)
    A = torch.tensor((Src), dtype = torch.float64)
    B = torch.tensor((Tar), dtype = torch.float64)

    if pca != PCAEnums.NO_PCA:
        # Override scaling to have no scaling bc we standardize before applying PCA
        scaling = ScalingEnums.NO_SCALING

    F1 = feature_extraction(Src1, features, scaling)
    F2 = feature_extraction(Tar1, features, scaling)
    combined_features = np.vstack((F1,F2))

    if pca != PCAEnums.NO_PCA:
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(combined_features)
        pca_obj = PCA(n_components=pca.value)
        principal_components = pca_obj.fit_transform(standardized_features)
        combined_features = principal_components

    if scaling == ScalingEnums.COLLECTIVE_STANDARDIZATION:
        # Standardization
        scaler = StandardScaler()
        combined_features = scaler.fit_transform(combined_features)

    if scaling == ScalingEnums.COLLECTIVE_MM_NORMALIZATION:
        # Min max normalization to 0-1 range
        scaler = MinMaxScaler()
        combined_features = scaler.fit_transform(combined_features)

    if scaling == ScalingEnums.COLLECTIVE_ROBUST_NORMALIZATION:
        scaler = RobustScaler()
        combined_features = scaler.fit_transform(combined_features)

    F1 = combined_features[:n,:]
    F2 = combined_features[n:,:]

    D = eucledian_dist(F1, F2, n)

    D = torch.tensor(D, dtype = torch.float64)
    print("source graph: ", Src[0, :10], " target graph: ", Tar[0, :10])
    
    P = convex_init(A, B, D, sinkhorn_reg, nu, mu, iter)

    print("The resulting souble stochastic matrix: ", P[0,:10])
    
    #P=convex_init1(A, B, L, mu, iter)
    #are_matrices_equal(P,P1)
    #P_perm, ans = convertToPermHungarian(P, n1, n2)
    return P
