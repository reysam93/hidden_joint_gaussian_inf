import numpy as np
import cvxpy as cp

def joint_inf_Apsd(Cos, regs):
    """
    Optimization problem solved with CVX to perform joint network topology inference
    in the presence of hidden variables when the observasions follow a Gaussian 
    distribution. The GSO is the adjacency matrix, which is estimated with a positive
    diagonal that will be set to 0 after the optimization is done.
    The input arguments are:
    - Cos: numpy matrix of size (N,O,O) corresponding to the observed Covariance 
           matrix of the K graphs
    - regs: dictionary with the weights for the regularizers. It should include:
        * rho1: weight for the l1 norm of $S^{(k)}$
        * rho2: weight for the l1 norm of $S^{(k)}-S^{(k')}$
        * beta1: weight for the nuclear norm of $P^{(k)}$
        * beta2: weight for the l1 norm of $P^{(k)}-P^{(k')}$
    """
    O = Cos.shape[1]
    K = Cos.shape[0]

    rho1 = regs['rho1']
    rho2 = regs['rho2']
    beta1 = regs['beta1']
    beta2 = regs['beta2']

    S_hat = []
    P_hat = []
    constraints = []
    for k in range(K):
        S_hat.append(cp.Variable((O, O), PSD=True))
        P_hat.append(cp.Variable((O, O), PSD=True))

        constraints += [S_hat[k] >= 0]
    
    constraints += [cp.sum(S_hat[0][1:,0]) == 1]
    
    non_diag = ~np.eye(O, dtype=bool)
    obj = 0
    for k in range(K):
        obj += cp.trace((S_hat[k] - P_hat[k])@Cos[k,:,:]) - cp.log_det(S_hat[k] - P_hat[k]) \
            + rho1*cp.norm(S_hat[k][non_diag], 1) + beta1*cp.norm(P_hat[k], 'nuc')

        for l in range(k):
            obj += rho2*cp.norm(S_hat[k][non_diag] - S_hat[l][non_diag], 1) \
                + beta2*cp.norm(P_hat[k] - P_hat[l], 1)

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print('WARNING: problem status', prob.status)
        return np.zeros(S_hat.shape), np.zeros(S_hat.shape)

    S_np = np.zeros((K, O , O))
    P_np = np.zeros((K, O , O))
    for k in range(K):
        S_np[k,:,:] = S_hat[k].value
        P_np[k,:,:] = P_hat[k].value

    return S_np, P_np


### BASELINES ###
def GL(Co, regs):
    O = Co.shape[0]

    rho1 = regs['rho1']

    S_hat = cp.Variable((O, O), PSD=True)

    contraints = [S_hat >= 0, cp.sum(S_hat[:,0]) == 1]
    non_diag = ~np.eye(O, dtype=bool)

    obj = cp.trace(S_hat@Co) - cp.log_det(S_hat) \
          + rho1*cp.norm(S_hat[non_diag], 1)


    prob = cp.Problem(cp.Minimize(obj), contraints)
    prob.solve()
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print('WARNING: problem status', prob.status)
        return np.zeros(S_hat.shape)

    return S_hat.value


def GGL(Cos, regs):
    """
    Optimization problem solved with CVX to perform joint network topology inference
    in the presence of hidden variables when the observasions follow a Gaussian 
    distribution. The GSO is the adjacency matrix, which is estimated with a positive
    diagonal that will be set to 0 after the optimization is done.
    The input arguments are:
    - Cos: numpy matrix of size (N,O,O) corresponding to the observed Covariance 
           matrix of the K graphs
    - regs: dictionary with the weights for the regularizers. It should include:
        * rho1: weight for the l1 norm of $S^{(k)}$
        * rho2: weight for the l1 norm of $S^{(k)}-S^{(k')}$
        * beta1: weight for the nuclear norm of $P^{(k)}$
        * beta2: weight for the l1 norm of $P^{(k)}-P^{(k')}$
    """
    O = Cos.shape[1]
    K = Cos.shape[0]

    rho1 = regs['rho1']
    rho2 = regs['rho2']

    S_hat = []
    constraints = []
    for k in range(K):
        S_hat.append(cp.Variable((O, O), PSD=True))
        constraints += [S_hat[k] >= 0]
    
    constraints += [cp.sum(S_hat[0][1:,0]) == 1]
    
    non_diag = ~np.eye(O, dtype=bool)
    obj = 0
    for k in range(K):
        obj += cp.trace(S_hat[k]@Cos[k,:,:]) - cp.log_det(S_hat[k]) \
            + rho1*cp.norm(S_hat[k][non_diag], 1)

        for l in range(k):
            obj += rho2*cp.norm(S_hat[k][non_diag] - S_hat[l][non_diag], 1)

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print('WARNING: problem status', prob.status)
        return np.zeros(S_hat.shape), np.zeros(S_hat.shape)

    S_np = np.zeros((K, O , O))
    for k in range(K):
        S_np[k,:,:] = S_hat[k].value

    return S_np


def LVGL(Co, regs):
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
