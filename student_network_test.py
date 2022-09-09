# This experiment evaluates the evolution of the error as the number of graphs to be inferred increases. It compares the performance of independently identifying K similar graphs with respect the performance of a joint inference algorithm.
# %% codecell
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

import time
from joblib import Parallel, delayed
from os import cpu_count

import utils
import opt

SEED = 0
N_CPUS = cpu_count()
np.random.seed(SEED)

# ### Auxiliary functions
def error_to_csv(fname, models, xaxis, error):
    header = ''
    data = error

    if xaxis is not None:
        data = np.concatenate((xaxis.reshape([xaxis.size, 1]), error.T), axis=1)
        header = 'xaxis, '

    for i, model in enumerate(models):
        header += model
        if i < len(models)-1:
            header += ', '

    np.savetxt(fname, data, delimiter=',', header=header, comments='')
    print('SAVED as:', fname)

def sep_inf_Apsd(Co, regs):
    O = Co.shape[0]

    rho1 = regs['rho1']
    beta1 = regs['beta1']

    S_hat = cp.Variable((O, O), PSD=True)
    P_hat = cp.Variable((O, O), PSD=True)

    contraints = [S_hat >= 0, cp.sum(S_hat[:,0]) == 1]
    non_diag = ~np.eye(O, dtype=bool)

    obj = cp.trace((S_hat - P_hat)@Co) - cp.log_det(S_hat - P_hat) \
          + rho1*cp.norm(S_hat[non_diag], 1) + beta1*cp.norm(P_hat, 'nuc')

    prob = cp.Problem(cp.Minimize(obj), contraints)
    prob.solve()
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print('WARNING: problem status', prob.status)
        return np.zeros(S_hat.shape), np.zeros(S_hat.shape)

    return S_hat.value, P_hat.value

def plot_err(KK, err, ylab, logy=True, ylim=[]):
    plt.figure()
    if logy:
        plt.semilogy(KK, err[:,0], 'o:', linewidth=2, markersize=12, label='GL')
        plt.semilogy(KK, err[:,1], 'o-.', linewidth=2, markersize=12, label='GGL')
        plt.semilogy(KK, err[:,2], 'o-', linewidth=2, markersize=12, label='LVGL')
        plt.semilogy(KK, err[:,3], 'o--', linewidth=2, markersize=12, label='Joint Hidden')
    else:
        plt.plot(KK, err[:,0], 'o:', linewidth=2, markersize=12, label='GL')
        plt.plot(KK, err[:,1], 'o-.', linewidth=2, markersize=12, label='GGL')
        plt.plot(KK, err[:,2], 'o-', linewidth=2, markersize=12, label='LVGL')
        plt.plot(KK, err[:,3], 'o--', linewidth=2, markersize=12, label='Joint Hidden')

    plt.grid(True)
    plt.xlabel('Number of graphs')
    plt.ylabel(ylab)
    plt.legend()
    plt.xlim([KK[0], KK[-1]])
    if ylim:
        plt.ylim(ylim)
    plt.tight_layout()

def run_stud_Mexp(id, K, N, O, MM, regs):
    regs_no_h = regs.copy()
    regs_no_h['beta1'] = 1e3
    regs_no_h['beta2'] = 0

    # Load graphs
    stud_folder = 'Student_Network_Data/'
    A_all = np.zeros((12,N,N))
    for i in range(12):
        net_list = np.loadtxt(stud_folder + 'as' + str(i+1) + '.net.txt').astype(int)
        for j in range(net_list.shape[0]):
            row = net_list[j,0]-1
            col = net_list[j,1]-1
            A_all[i,row,col] = A_all[i,col,row] = 1
    A_all = np.array(A_all)
    As = A_all[np.random.permutation(12)[:K],:,:]

    # Fix scale
    As = As/np.sum(As[0,1:,0])

    Cs_hat = np.zeros((len(MM),K,N,N))
    for m in range(len(MM)):
        _, Cs_curr = utils.create_GMRF_data(As, MM[m])
        Cs_hat[m,:,:,:] = Cs_curr

    # Randomly select observed/hidden nodes
    rand_idx = np.random.permutation(np.arange(N))
    idx_o = rand_idx[:O]

    Aos = As[:,idx_o,:][:,:,idx_o]
    Cos_hat = Cs_hat[:,:,idx_o,:][:,:,:,idx_o]

    diag_idx = np.eye(O, dtype=bool)
    err = np.zeros((len(MM), 4))
    err_mean_norm = np.zeros((len(MM), 4))
    err_sum_norm = np.zeros((len(MM), 4))
    for i, M in enumerate(MM):
        Cok_hat = Cos_hat[i,:,:,:]

        S_ggl, _ = opt.joint_inf_Apsd(Cok_hat, regs_no_h)
        S_joint, _ = opt.joint_inf_Apsd(Cok_hat, regs)

        norm_Ao_sq = 0
        S_lvgl = np.zeros(Cok_hat.shape)
        S_sep_noh = np.zeros(Cok_hat.shape)
        errs_aux = np.zeros(4)
        for k in range(K):
            norm_Aok = np.linalg.norm(Aos[k,:,:], 'fro')
            norm_Ao_sq += norm_Aok**2
            S_lvgl[k,:,:], _ = sep_inf_Apsd(Cok_hat[k,:,:], regs)
            S_sep_noh[k,:,:], _ = sep_inf_Apsd(Cok_hat[k,:,:], regs_no_h)

            # Set diags to 0
            S_sep_noh[k,:,:][diag_idx] = 0
            S_ggl[k,:,:][diag_idx] = 0
            S_lvgl[k,:,:][diag_idx] = 0
            S_joint[k,:,:][diag_idx] = 0

            # Errs
            errs_aux[0] = np.linalg.norm(S_sep_noh[k,:,:] - Aos[k,:,:], 'fro')**2
            errs_aux[1] = np.linalg.norm(S_ggl[k,:,:] - Aos[k,:,:], 'fro')**2
            errs_aux[2] = np.linalg.norm(S_lvgl[k,:,:] - Aos[k,:,:], 'fro')**2
            errs_aux[3] = np.linalg.norm(S_joint[k,:,:] - Aos[k,:,:], 'fro')**2

            # Compute errors
            err[i,:] += errs_aux
            err_mean_norm[i,:] += errs_aux/norm_Aok**2/K

        err_sum_norm[i,:] = err[i,:]/norm_Ao_sq

        print('{}-{}: Err LVGL: {:.4f} - Err joint: {:.4f}'.format(id, K, err_mean_norm[i,2], err_mean_norm[i,3]))

    # return err_sep, err_joint, err_sep_norm, err_joint_norm
    return err, err_mean_norm, err_sum_norm

def run_stud_Oexp(id, K, N, OO, M, regs):
    regs_no_h = regs.copy()
    regs_no_h['beta1'] = 1e3
    regs_no_h['beta2'] = 0

    # Load graphs
    stud_folder = 'Student_Network_Data/'
    A_all = np.zeros((12,N,N))
    for i in range(12):
        net_list = np.loadtxt(stud_folder + 'as' + str(i+1) + '.net.txt').astype(int)
        for j in range(net_list.shape[0]):
            row = net_list[j,0]-1
            col = net_list[j,1]-1
            A_all[i,row,col] = A_all[i,col,row] = 1
    A_all = np.array(A_all)
    As = A_all[np.random.permutation(12)[:K],:,:]

    # Fix scale
    As = As/np.sum(As[0,1:,0])

    _, Cs_hat = utils.create_GMRF_data(As, M)

    # Randomly select observed/hidden nodes
    rand_idx = np.random.permutation(np.arange(N))
    err = np.zeros((len(OO), 4))
    err_mean_norm = np.zeros((len(OO), 4))
    err_sum_norm = np.zeros((len(OO), 4))

    for i, O in enumerate(OO):
        idx_o = rand_idx[:O]

        Aos = As[:,idx_o,:][:,:,idx_o]
        Cos_hat = Cs_hat[:,idx_o,:][:,:,idx_o]

        diag_idx = np.eye(O, dtype=bool)
        Cok_hat = Cos_hat[:K,:,:]

        S_ggl, _ = opt.joint_inf_Apsd(Cok_hat, regs_no_h)
        S_joint, _ = opt.joint_inf_Apsd(Cok_hat, regs)

        norm_Ao_sq = 0
        S_lvgl = np.zeros(Cok_hat.shape)
        S_sep_noh = np.zeros(Cok_hat.shape)
        errs_aux = np.zeros(4)
        for k in range(K):
            norm_Aok = np.linalg.norm(Aos[k,:,:], 'fro')
            norm_Ao_sq += norm_Aok**2
            S_lvgl[k,:,:], _ = sep_inf_Apsd(Cok_hat[k,:,:], regs)
            S_sep_noh[k,:,:], _ = sep_inf_Apsd(Cok_hat[k,:,:], regs_no_h)

            # Set diags to 0
            S_sep_noh[k,:,:][diag_idx] = 0
            S_ggl[k,:,:][diag_idx] = 0
            S_lvgl[k,:,:][diag_idx] = 0
            S_joint[k,:,:][diag_idx] = 0

            # Errs
            errs_aux[0] = np.linalg.norm(S_sep_noh[k,:,:] - Aos[k,:,:], 'fro')**2
            errs_aux[1] = np.linalg.norm(S_ggl[k,:,:] - Aos[k,:,:], 'fro')**2
            errs_aux[2] = np.linalg.norm(S_lvgl[k,:,:] - Aos[k,:,:], 'fro')**2
            errs_aux[3] = np.linalg.norm(S_joint[k,:,:] - Aos[k,:,:], 'fro')**2

            # Compute errors
            err[i,:] += errs_aux
            err_mean_norm[i,:] += errs_aux/norm_Aok**2/K

        err_sum_norm[i,:] = err[i,:]/norm_Ao_sq

        print('{}-{}: Err LVGL: {:.4f} - Err joint: {:.4f}'.format(id, K, err_mean_norm[i,2], err_mean_norm[i,3]))

    # return err_sep, err_joint, err_sep_norm, err_joint_norm
    return err, err_mean_norm, err_sum_norm

def run_stud_Kexp(id, KK, N, O, M, regs):
    regs_no_h = regs.copy()
    regs_no_h['beta1'] = 1e3
    regs_no_h['beta2'] = 0

    # Load graphs
    stud_folder = 'Student_Network_Data/'
    A_all = np.zeros((12,N,N))
    for i in range(12):
        net_list = np.loadtxt(stud_folder + 'as' + str(i+1) + '.net.txt').astype(int)
        for j in range(net_list.shape[0]):
            row = net_list[j,0]-1
            col = net_list[j,1]-1
            A_all[i,row,col] = A_all[i,col,row] = 1
    A_all = np.array(A_all)
    As = A_all[:KK[-1],:,:]

    # Fix scale
    As = As/np.sum(As[0,1:,0])

    _, Cs_hat = utils.create_GMRF_data(As, M)

    # Randomly select observed/hidden nodes
    rand_idx = np.random.permutation(np.arange(N))
    idx_o = rand_idx[:O]

    Aos = As[:,idx_o,:][:,:,idx_o]
    Cos_hat = Cs_hat[:,idx_o,:][:,:,idx_o]

    diag_idx = np.eye(O, dtype=bool)
    err = np.zeros((len(KK), 4))
    err_mean_norm = np.zeros((len(KK), 4))
    err_sum_norm = np.zeros((len(KK), 4))
    for i, K in enumerate(KK):
        Cok_hat = Cos_hat[:K,:,:]

        S_ggl, _ = opt.joint_inf_Apsd(Cok_hat, regs_no_h)
        S_joint, _ = opt.joint_inf_Apsd(Cok_hat, regs)

        norm_Ao_sq = 0
        S_lvgl = np.zeros(Cok_hat.shape)
        S_sep_noh = np.zeros(Cok_hat.shape)
        errs_aux = np.zeros(4)
        for k in range(K):
            norm_Aok = np.linalg.norm(Aos[k,:,:], 'fro')
            norm_Ao_sq += norm_Aok**2
            S_lvgl[k,:,:], _ = sep_inf_Apsd(Cok_hat[k,:,:], regs)
            S_sep_noh[k,:,:], _ = sep_inf_Apsd(Cok_hat[k,:,:], regs_no_h)

            # Set diags to 0
            S_sep_noh[k,:,:][diag_idx] = 0
            S_ggl[k,:,:][diag_idx] = 0
            S_lvgl[k,:,:][diag_idx] = 0
            S_joint[k,:,:][diag_idx] = 0

            # Errs
            errs_aux[0] = np.linalg.norm(S_sep_noh[k,:,:] - Aos[k,:,:], 'fro')**2
            errs_aux[1] = np.linalg.norm(S_ggl[k,:,:] - Aos[k,:,:], 'fro')**2
            errs_aux[2] = np.linalg.norm(S_lvgl[k,:,:] - Aos[k,:,:], 'fro')**2
            errs_aux[3] = np.linalg.norm(S_joint[k,:,:] - Aos[k,:,:], 'fro')**2

            # Compute errors
            err[i,:] += errs_aux
            err_mean_norm[i,:] += errs_aux/norm_Aok**2/K

        err_sum_norm[i,:] = err[i,:]/norm_Ao_sq

        print('{}-{}: Err LVGL: {:.4f} - Err joint: {:.4f}'.format(id, K, err_mean_norm[i,2], err_mean_norm[i,3]))

    # return err_sep, err_joint, err_sep_norm, err_joint_norm
    return err, err_mean_norm, err_sum_norm

def run_exp(id, KK, N, O, p, M, pert_links, regs):
    regs_no_h = regs.copy()
    regs_no_h['beta1'] = 1e3
    regs_no_h['beta2'] = 0

    # Create graphs
    A = np.zeros((N,N))
    tries = 0
    while np.any(np.sum(A,axis=1) == 0) and tries < 10:
        tries += 1
        A = nx.to_numpy_array(nx.erdos_renyi_graph(N, p))

    assert not np.any(np.sum(A,axis=1) == 0), 'Graphs with nodes with degree 0'

    As = utils.gen_similar_graphs(A, KK[-1], pert_links)

    # Fix scale
    As = As/np.sum(As[0,1:,0])

    _, Cs_hat = utils.create_GMRF_data(As, M)

    # Randomly select observed/hidden nodes
    rand_idx = np.random.permutation(np.arange(N))
    idx_o = rand_idx[:O]

    Aos = As[:,idx_o,:][:,:,idx_o]
    Cos_hat = Cs_hat[:,idx_o,:][:,:,idx_o]

    diag_idx = np.eye(O, dtype=bool)
    err = np.zeros((len(KK), 4))
    err_mean_norm = np.zeros((len(KK), 4))
    err_sum_norm = np.zeros((len(KK), 4))
    for i, K in enumerate(KK):
        Cok_hat = Cos_hat[:K,:,:]

        S_ggl, _ = opt.joint_inf_Apsd(Cok_hat, regs_no_h)
        S_joint, _ = opt.joint_inf_Apsd(Cok_hat, regs)

        norm_Ao_sq = 0
        S_lvgl = np.zeros(Cok_hat.shape)
        S_sep_noh = np.zeros(Cok_hat.shape)
        errs_aux = np.zeros(4)
        for k in range(K):
            norm_Aok = np.linalg.norm(Aos[k,:,:], 'fro')
            norm_Ao_sq += norm_Aok**2
            S_lvgl[k,:,:], _ = sep_inf_Apsd(Cok_hat[k,:,:], regs)
            S_sep_noh[k,:,:], _ = sep_inf_Apsd(Cok_hat[k,:,:], regs_no_h)

            # Set diags to 0
            S_sep_noh[k,:,:][diag_idx] = 0
            S_ggl[k,:,:][diag_idx] = 0
            S_lvgl[k,:,:][diag_idx] = 0
            S_joint[k,:,:][diag_idx] = 0

            # Errs
            errs_aux[0] = np.linalg.norm(S_sep_noh[k,:,:] - Aos[k,:,:], 'fro')**2
            errs_aux[1] = np.linalg.norm(S_ggl[k,:,:] - Aos[k,:,:], 'fro')**2
            errs_aux[2] = np.linalg.norm(S_lvgl[k,:,:] - Aos[k,:,:], 'fro')**2
            errs_aux[3] = np.linalg.norm(S_joint[k,:,:] - Aos[k,:,:], 'fro')**2

            # Compute errors
            err[i,:] += errs_aux
            err_mean_norm[i,:] += errs_aux/norm_Aok**2/K

        err_sum_norm[i,:] = err[i,:]/norm_Ao_sq

        print('{}-{}: Err LVGL: {:.4f} - Err joint: {:.4f}'.format(id, K, err_mean_norm[i,2], err_mean_norm[i,3]))

    # return err_sep, err_joint, err_sep_norm, err_joint_norm
    return err, err_mean_norm, err_sum_norm

# %% codecell
# Experiment parameters
K = 3
N = 32
O = 31
MM = np.logspace(2,5,7).astype(int)
n_graphs = 25
regs = {'rho1': .01, 'rho2': .01, 'beta1': .1, 'beta2': .1}

err = np.zeros((len(MM), 4, n_graphs))
err_mean_norm = np.zeros((len(MM), 4, n_graphs))
err_sum_norm = np.zeros((len(MM), 4, n_graphs))

total_t = time.time()
print('N_CPUS:', N_CPUS)
pool = Parallel(n_jobs=N_CPUS, verbose=0)
resps = pool(delayed(run_stud_Mexp)(i, K, N, O, MM, regs) for i in range(n_graphs))
for i, resp in enumerate(resps):
    err[:,:,i], err_mean_norm[:,:,i], err_sum_norm[:,:,i] = resp

total_t = time.time() - total_t
print('-----', total_t/60, ' mins -----')

err_mean = np.mean(err, axis=2)
err_mean_norm_mean = np.mean(err_mean_norm, axis=2)
err_sum_norm_mean = np.mean(err_sum_norm, axis=2)

print(err_mean)
print(err_mean_norm_mean)
print(err_sum_norm_mean)

plot_err(MM, err_mean, 'Mean err', logy=False)
plot_err(MM, err_mean_norm_mean, 'Mean err', logy=False)
plot_err(MM, err_sum_norm_mean, 'Mean err', logy=False)

err_mean = np.mean(err, axis=2)
err_mean_norm_mean = np.mean(err_mean_norm, axis=2)
err_sum_norm_mean = np.mean(err_sum_norm, axis=2)

print(err_mean)
print(err_mean_norm_mean)
print(err_sum_norm_mean)

plot_err(MM, err_mean, 'Mean err', logy=True)
plot_err(MM, err_mean_norm_mean, 'Mean err', logy=True)
plot_err(MM, err_sum_norm_mean, 'Mean err', logy=True)


error_to_csv('data_Mexp_mean',['GL','GGL','LVGL','Joint'],np.array(MM),err_mean_norm_mean.T)
error_to_csv('data_Mexp_sum',['GL','GGL','LVGL','Joint'],np.array(MM),err_sum_norm_mean.T)

# %% codecell
# Experiment parameters
K = 4
N = 32
OO = 32-np.arange(1,12,2)
M = 200
n_graphs = 25
regs = {'rho1': .01, 'rho2': .01, 'beta1': .1, 'beta2': .1}

err = np.zeros((len(OO), 4, n_graphs))
err_mean_norm = np.zeros((len(OO), 4, n_graphs))
err_sum_norm = np.zeros((len(OO), 4, n_graphs))

total_t = time.time()
print('N_CPUS:', N_CPUS)
pool = Parallel(n_jobs=N_CPUS, verbose=0)
resps = pool(delayed(run_stud_Oexp)(i, K, N, OO, M, regs) for i in range(n_graphs))
for i, resp in enumerate(resps):
    err[:,:,i], err_mean_norm[:,:,i], err_sum_norm[:,:,i] = resp

total_t = time.time() - total_t
print('-----', total_t/60, ' mins -----')
err_mean = np.mean(err, axis=2)
err_mean_norm_mean = np.mean(err_mean_norm, axis=2)
err_sum_norm_mean = np.mean(err_sum_norm, axis=2)

print(err_mean)
print(err_mean_norm_mean)
print(err_sum_norm_mean)

plot_err(OO, err_mean, 'Mean err', logy=False)
plot_err(OO, err_mean_norm_mean, 'Mean err', logy=False)
plot_err(OO, err_sum_norm_mean, 'Mean err', logy=False)

err_mean = np.mean(err, axis=2)
err_mean_norm_mean = np.mean(err_mean_norm, axis=2)
err_sum_norm_mean = np.mean(err_sum_norm, axis=2)

print(err_mean)
print(err_mean_norm_mean)
print(err_sum_norm_mean)

plot_err(OO, err_mean, 'Mean err', logy=True)
plot_err(OO, err_mean_norm_mean, 'Mean err', logy=True)
plot_err(OO, err_sum_norm_mean, 'Mean err', logy=True)

error_to_csv('data_Oexp_mean',['GL','GGL','LVGL','Joint'],np.array(OO),err_mean_norm_mean.T)
error_to_csv('data_Oexp_sum',['GL','GGL','LVGL','Joint'],np.array(OO),err_sum_norm_mean.T)

# %% codecell
# Experiment parameters
KK = [1, 2, 3, 4, 5, 6]
N = 32
O = 31
p = .15
M = 200
pert_links = 5
n_graphs = 25
regs = {'rho1': .01, 'rho2': .01, 'beta1': .1, 'beta2': .1}

# (K=1) Min mean err (rho1: 0.01, rho2: 0, beta1: 0.1, beta2: 0): 0.1466
# (K=3) Min mean err (rho1: 0.01, rho2: 0.01, beta1: 0.1, beta2: 0.01): 0.1072
# (K=3) Min mean err (rho1: 0.01, rho2: 0.01, beta1: 0.1, beta2: 0.01): 0.0871
# (K=5) Min mean err (rho1: 0.01, rho2: 0.01, beta1: 0.1, beta2: 0.1): 0.0814
# (K=7) Min mean err (rho1: 0.01, rho2: 0.01, beta1: 0.1, beta2: 0.1): 0.0689

err = np.zeros((len(KK), 4, n_graphs))
err_mean_norm = np.zeros((len(KK), 4, n_graphs))
err_sum_norm = np.zeros((len(KK), 4, n_graphs))

total_t = time.time()
print('N_CPUS:', N_CPUS)
pool = Parallel(n_jobs=N_CPUS, verbose=0)
resps = pool(delayed(run_stud_Kexp)(i, KK, N, O, M, regs) for i in range(n_graphs))
for i, resp in enumerate(resps):
    err[:,:,i], err_mean_norm[:,:,i], err_sum_norm[:,:,i] = resp

total_t = time.time() - total_t
print('-----', total_t/60, ' mins -----')

err_mean = np.mean(err, axis=2)
err_mean_norm_mean = np.mean(err_mean_norm, axis=2)
err_sum_norm_mean = np.mean(err_sum_norm, axis=2)

print(err_mean)
print(err_mean_norm_mean)
print(err_sum_norm_mean)

plot_err(KK, err_mean, 'Mean err', logy=False)
plot_err(KK, err_mean_norm_mean, 'Mean err', logy=False)
plot_err(KK, err_sum_norm_mean, 'Mean err', logy=False)
# %% codecell
err_mean = np.mean(err, axis=2)
err_mean_norm_mean = np.mean(err_mean_norm, axis=2)
err_sum_norm_mean = np.mean(err_sum_norm, axis=2)

print(err_mean)
print(err_mean_norm_mean)
print(err_sum_norm_mean)

plot_err(KK, err_mean, 'Mean err', logy=True)
plot_err(KK, err_mean_norm_mean, 'Mean err', logy=True)
plot_err(KK, err_sum_norm_mean, 'Mean err', logy=True)

error_to_csv('data_Kexp_mean',['GL','GGL','LVGL','Joint'],np.array(KK),err_mean_norm_mean.T)
error_to_csv('data_Kexp_sum',['GL','GGL','LVGL','Joint'],np.array(KK),err_sum_norm_mean.T)
