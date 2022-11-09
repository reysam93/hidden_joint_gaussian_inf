import networkx as nx
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

import utils
import opt

import time
from joblib import Parallel, delayed
from os import cpu_count

STUD_FOLDER = 'Student_Network_Data/'
STUD_GRAPHS = 12
SEED = 0
MAX_TRIES = 25
N_CPUS = cpu_count()
np.random.seed(SEED)

MODELS = ['Joint', 'GL', 'GGL', 'LVGL']

G_PARAMS = [{'type': 'ER', 'N': 20, 'p': .15},
            {'type': 'STUDENT'}]

def similar_ER_graphs(params, pert_links):
    N = params['N']
    p = params['p']

    A = np.zeros((N,N))
    tries = 0
    while np.any(np.sum(A,axis=1) == 0) and tries < MAX_TRIES:
        tries += 1
        A = nx.to_numpy_array(nx.erdos_renyi_graph(N, p))

    assert not np.any(np.sum(A,axis=1) == 0), 'Graphs with 0 nodes'

    As = utils.gen_similar_graphs(A, K, pert_links)
    return As


def load_student_graphs(N):
    A_all = np.zeros((STUD_GRAPHS, N, N))
    for i in range(STUD_GRAPHS):
        net_list = np.loadtxt(STUD_FOLDER + 'as' + str(i+1) + '.net.txt').astype(int)
        for j in range(net_list.shape[0]):
            row = net_list[j,0]-1
            col = net_list[j,1]-1
            A_all[i,row,col] = A_all[i,col,row] = 1
    A_all = np.array(A_all)
    As = A_all[np.random.permutation(STUD_GRAPHS)[:K],:,:]
    return As


def run_exp(id, model, K, g_params, H, M, pert_links):
    assert model in MODELS, f'Model must be one of {MODELS}'
    assert g_params in G_PARAMS, f'Graph parameters must be in {G_PARAMS}'

    # Create data
    if g_params['type'] == 'STUDENT':
        N = 32
        As = load_student_graphs(N)
    else:
        As = similar_ER_graphs(g_params, pert_links)
        N = As.shape[1]

    # Fix scale
    As = As/np.sum(As[0,1:,0])
    O = N - H
    _, Cs_hat = utils.create_GMRF_data(As, M)

    # Randomly select observed/hidden nodes
    rand_idx = np.random.permutation(np.arange(N))
    idx_o = rand_idx[:O]
    Aos = As[:,idx_o,:][:,:,idx_o]
    Cos_hat = Cs_hat[:,idx_o,:][:,:,idx_o]

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

                    # Estimate graph according to selected model
                    if model == 'Joint':
                        S_joint, _ = opt.joint_inf_Apsd(Cos_hat, regs)
                    elif model == 'GGL':
                        S_joint = opt.GGL(Cos_hat, regs)
                    else:
                        S_joint = np.zeros((K, O, O))
                        for k in range(K):
                            if model == 'GL':
                                S_joint[k,:,:] = opt.GL(Cos_hat[k,:,:], regs)
                            elif model == 'LVGL':
                                S_joint[k,:,:], _ = opt.LVGL(Cos_hat[k,:,:], regs)
                        

                    # COMPUTE ERROR                  
                    for k in range(K):
                        norm_Ao = la.norm(Aos[k,:,:], 'fro')
                        S_joint[k,:,:][diag_idx] = 0
                        err[i,j,g,l] += (la.norm(S_joint[k,:,:] - Aos[k,:,:], 'fro')/norm_Ao)**2/K

                    print('G-{}: rho1 {}, rho2 {}, beta1 {}, beta2 {}: Err: {:.4f}'.
                          format(id, rho1, rho2, beta1, beta2, err[i,j,g,l]))

    return err

if __name__ == "__main__":
    np.random.seed(SEED)
    rhos1 = [.01, .05, .1, .5] # [.001, .01, .1]
    rhos2 = [0]
    betas1 = [.01, .1, 1, 2.5]
    betas2 = [0]  # [.01, .1, .5]

# For GL
# Min mean err (rho1: 0.1, rho2: 0, beta1: 0, beta2: 0): 0.2450

# For GGL
# Min mean err (rho1: 0.01, rho2: 0.1, beta1: 0, beta2: 0): 0.1960

# For LVGL
# Min mean err (rho1: 0.05, rho2: 0, beta1: 1, beta2: 0): 0.2188

    n_graphs = 10
    H = 2
    M = 200
    K = 4
    pert_links = 5

    model = MODELS[3]
    g_params = G_PARAMS[1]

    t = time.time()
    print("CPUs used:", N_CPUS, ', model:', model)
    err = np.zeros((len(rhos1), len(rhos2), len(betas1), len(betas2), n_graphs))

    pool = Parallel(n_jobs=N_CPUS, verbose=0)
    resps = pool(delayed(run_exp)(i, model, K, g_params, H, M, pert_links) for i in range(n_graphs))
    for i, resp in enumerate(resps):
        err[:,:,:,:,i] = resp
    print('----- {} mins -----'.format((time.time()-t)/60))

    mean_err = np.mean(err, 4)
    med_err = np.median(err, 4)
    
    # Print mean err
    idx = np.unravel_index(np.argmin(mean_err), mean_err.shape)
    print('Min mean err (rho1: {:.3g}, rho2: {:.3g}, beta1: {:.3g}, beta2: {:.3g}): {:.4f}'
        .format(rhos1[idx[0]], rhos2[idx[1]], betas1[idx[2]], betas2[idx[3]], mean_err[idx]))

    print()

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


    np.save('params_tmp', err)