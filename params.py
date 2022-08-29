import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

import utils

import time
from joblib import Parallel, delayed
from os import cpu_count

SEED = 0
MAX_TRIES = 25
N_CPUS = cpu_count()
np.random.seed(SEED)


def run_exp(id, K, N, O, p, M, pert_links):
    # Create data
    A = np.zeros((N,N))
    tries = 0
    while np.any(np.sum(A,axis=1) == 0) and tries < MAX_TRIES:
        tries += 1
        A = nx.to_numpy_array(nx.erdos_renyi_graph(N, p))

    assert not np.any(np.sum(A,axis=1) == 0), 'Graphs with 0 nodes'

    As = utils.gen_similar_graphs(A, K, pert_links)
    # Fix scale
    As = As/np.sum(As[0,1:,0])

    _, Cs_hat = utils.create_GMRF_data(As, M)

    # Randomly select observed/hidden nodes
    rand_idx = np.random.permutation(np.arange(N))
    idx_o = rand_idx[:O]
    Aos = As[:,idx_o,:][:,:,idx_o]
    Cos_hat = Cs_hat[:,idx_o,:][:,:,idx_o]

    norm_Ao = 0
    for k in range(K):
        norm_Ao += np.linalg.norm(Aos[k,:,:], 'fro')**2

    regs = {}
    diag_idx = np.eye(O, dtype=bool)
    err = np.zeros((len(rhos1), len(rhos2), len(betas1), len(betas2)))
    for l, beta2 in enumerate(betas2):
        regs['beta2'] = beta2
        for g, beta1 in enumerate(betas1):
            regs['beta1'] = beta1
            for j, rho2 in enumerate(rhos2):
                regs['rho2'] = rho2
                for i, rho1 in enumerate(rhos1):
                    regs['rho1'] = rho1

                    S_joint, _ = utils.joint_inf_Apsd(Cos_hat, regs)                    
                    for k in range(K):
                        S_joint[k,:,:][diag_idx] = 0
                        err[i,j,g,l] += np.linalg.norm(S_joint[k,:,:] - Aos[k,:,:], 'fro')**2
                    err[i,j,g,l] =  err[i,j,g,l]/norm_Ao

                    print('G-{}: rho1 {}, rho2 {}, beta1 {}, beta2 {}: Err: {:.4f}'.
                          format(id, rho1, rho2, beta1, beta2, err[i,j,g,l]))

    return err

if __name__ == "__main__":
    np.random.seed(SEED)
    rhos1 = [.001, .01, .1]
    rhos2 =  [.001, .01, .1, .5]
    betas1 = [.01, .1, .5]
    betas2 = [.01, .1, .5]


    n_graphs = 10
    N = 20
    O = 19
    p = .15
    M = 200
    K = 7
    pert_links = 5

    t = time.time()
    print("CPUs used:", N_CPUS)
    err = np.zeros((len(rhos1), len(rhos2), len(betas1), len(betas2), n_graphs))

    pool = Parallel(n_jobs=N_CPUS, verbose=0)
    resps = pool(delayed(run_exp)(i, K, N, O, p, M, pert_links) for i in range(n_graphs))
    for i, resp in enumerate(resps):
        err[:,:,:,:,i] = resp
    print('----- {} mins -----'.format((time.time()-t)/60))

    mean_err = np.mean(err, 4)
    med_err = np.median(err, 4)
    
    # Print mean err
    idx = np.unravel_index(np.argmin(mean_err), mean_err.shape)
    print('Min mean err (rho1: {:.3g}, rho2: {:.3g}, beta1: {:.3g}, beta2: {:.3g}): {:.4f}'
        .format(rhos1[idx[0]], rhos2[idx[1]], betas1[idx[2]], betas2[idx[3]], mean_err[idx]))

    # Print median err
    idx = np.unravel_index(np.argmin(med_err), med_err.shape)
    print('Min median err (rho1: {:.3g}, rho2: {:.3g}, beta1: {:.3g}, beta2: {:.3g}): {:.4f}'
        .format(rhos1[idx[0]], rhos2[idx[1]], betas1[idx[2]], betas2[idx[3]], med_err[idx]))

    
    # # Print mean err - separate
    # idx = np.unravel_index(np.argmin(mean_err[:,0,:,0]), mean_err[:,0,:,0].shape)
    # print('Min err (rho1: {:.3g}, rho2: {:.3g}, beta1: {:.3g}, beta2: {:.3g}): {:.4f}'
    #     .format(rhos1[idx[0]], 0, betas1[idx[2]], 0, mean_err[idx]))

    # # Print median err - separate
    # idx = np.unravel_index(np.argmin(med_err[:,0,:,0]), med_err[:,0,:,0].shape)
    # print('Min err (rho1: {:.3g}, rho2: {:.3g}, beta1: {:.3g}, beta2: {:.3g}): {:.4f}'
    #     .format(rhos1[idx[0]], 0, betas1[idx[2]], 0, med_err[idx]))



    print(mean_err)

    np.save('params_tmp', err)