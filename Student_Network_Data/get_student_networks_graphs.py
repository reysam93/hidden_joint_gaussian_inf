import numpy as np

def get_student_networks_graphs(graphs_idx, path):
    assert max(graphs_idx) < 12 and min(graphs_idx) > 0, \
        'graph_idx should be a ector with values between 1 and 12'

    K = len(graphs_idx)
    Gs = []
    for k in range(K):
        file_path = '{}as{}.net.txt'.format(path, graphs_idx[k])
        Gs.append(np.loadtxt(file_path))

    N = int(Gs[0].max())
    As = np.zeros((N, N, K))
    for k in range(K):
        for idx in range(Gs[k].shape[0]):
            i = int(Gs[k][idx,0]) - 1
            j = int(Gs[k][idx,1]) - 1
            weight = Gs[k][idx,2]
            As[i,j,k] = weight

    return As

get_student_networks_graphs([2], './')
print('done')