import numpy as np


def gen_similar_graphs(A, K, pert_links):
    """ Create K similargraphs by rewiring edges """ 
    N = A.shape[0]
    As = np.zeros((K, N, N))
    As[0,:,:] = A
    for k in range(1, K):
        As[k,:,:] = A

        # Remove this for loop?
        for _ in range(pert_links):
            node_id = np.random.randint(N)

            # delete link
            link_nodes = np.where(As[0,:,node_id] != 0)[0]
            del_node = np.random.permutation(link_nodes)[0]
            As[k, node_id, del_node] = As[k, del_node, node_id] = 0

            # create link
            nonlink_nodes = np.where(As[0,:,node_id] == 0)[0]
            nonlink_nodes = np.delete(nonlink_nodes, np.where(nonlink_nodes == node_id))  # avoid self loops
            add_node = np.random.permutation(nonlink_nodes)[0]
            As[k, node_id, add_node] = As[k, add_node, node_id] = 1

        assert np.allclose(As[k], As[k].T), 'Non-symmetric matrix'
        assert np.sum(np.diag(As[k])) == 0, 'Non-zero diagonal'

    return As


def create_GMRF_data(As, M):
    """" Create covarianze matrix by converting A to a possitive definite matrix """
    Cs = np.zeros(As.shape)
    Cs_hat = np.zeros(As.shape)
    for k in range(As.shape[0]):
        eigvals = np.linalg.eigvalsh(As[k])
        C_inv = (.01-eigvals.min())*np.eye(As.shape[1]) + ((.9+.1*np.random.rand(1))*As[k])
        Cs[k] = np.linalg.inv(C_inv)

        # Returned X has a shape MxNxK
        X = np.random.multivariate_normal(np.zeros(Cs.shape[1]), Cs[k], M)
        Cs_hat[k] = X.T@X/M  
    return Cs, Cs_hat