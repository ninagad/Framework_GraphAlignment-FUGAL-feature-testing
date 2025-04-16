from algorithms.cuGAL.cugal.pred import cugal
from algorithms.cuGAL.cugal.config import Config, SinkhornMethod, HungarianMethod
from algorithms.cuGAL.cugal.profile import Profile, append_phases_to_csv
#from fugal.pred import fugal
import numpy as np
import networkx as nx
import torch
from enums.pcaEnums import PCAEnums
from enums.scalingEnums import ScalingEnums

def main(data, 
         iter,
         mu,
         nu: float,
         features: list,
         scaling: ScalingEnums,
         pca: PCAEnums,
         path=None, 
         sparse=False, 
         cache=0, 
         sinkhorn_method=SinkhornMethod.LOG,
         sinkhorn_iterations=500,
         sinkhorn_reg = 1,
         frank_wolfe_threshold=None,
         hungarian=HungarianMethod.SCIPY,
         device='cuda:0',
         dtype=torch.float32,
         sinkhorn_threshold=None,
         use_fugal=False,
         lambda_func=lambda x: x,
         ):
    
    if sinkhorn_threshold is None:
        sinkhorn_threshold = torch.finfo(dtype).eps

    config = Config(
        device=device, 
        sinkhorn_method=sinkhorn_method, 
        dtype=dtype,
        sinkhorn_threshold=sinkhorn_threshold,
        sinkhorn_iterations=sinkhorn_iterations,
        iter_count=iter,
        mu=mu,
        nu=nu,
        scaling=scaling,
        pca=pca,
        use_sparse_adjacency=sparse,
        #sinkhorn_cache_size=cache,
        sinkhorn_regularization=sinkhorn_reg,
        frank_wolfe_threshold=frank_wolfe_threshold,
        recompute_distance=True,
        hungarian_method=hungarian,
        #lambda_func=lambda_func,
    )

    print("nu: ", nu)

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
    Src1 = nx.from_numpy_array(Src)
    Tar1 = nx.from_numpy_array(Tar)

    if pca != PCAEnums.NO_PCA:
        # Override scaling to have no scaling bc we standardize before applying PCA
        scaling = ScalingEnums.NO_SCALING

    profile = Profile()

    #if use_fugal:
    #    P, mapping = fugal(Src1, Tar1, mu, iter, config, profile)
    #else:
    P, mapping = cugal(Src1, Tar1, features, scaling, pca, config, profile)
    #print("Sinkhorn threshold: ", config.sinkhorn_threshold)
    #print("Max Sinkhorn iterations: ", np.max([sinkhorn_profile.iteration_count for sinkhorn_profile in profile.sinkhorn_profiles]))
    #print("Mean Sinkhorn iterations: ", np.mean([sinkhorn_profile.iteration_count for sinkhorn_profile in profile.sinkhorn_profiles]))
    #print("Max memory used: ", profile.max_memory)

    if not path == None: 
        append_phases_to_csv(profile, path)
        #for sp in profile.sinkhorn_profiles:
            #if not sp.res_matrix is None:
            #    np.savetxt(path + 'sinkhorn_' + str(sp.iteration_count) + '.csv', sp.res_matrix.numpy(), delimiter=',')

    return P
