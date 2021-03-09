import math

__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import torch

def kmeans(
        X,
        num_clusters,
        cluster_centers,
        allow_distances,
        distance='euclidean',
        tol=1e-4,
        device=torch.device('cpu')):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param allow_distances: allow max distance
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # transfer ndarray to tensor
    allow_distances = torch.from_numpy(allow_distances).to(device)

    # find data point closest to the initial cluster center
    initial_state = cluster_centers.to(device)

    iteration = 0
    # tqdm_meter = tqdm(desc='[running kmeans]')
    while True:
        dis = pairwise_distance_function(X, initial_state, device)
        min_distance, choice_cluster = torch.min(dis, dim=1)

        # set cluster id = num_cluster if min distance is greater than allow distance
        allow_mask = (min_distance < allow_distances[choice_cluster]).long()
        choice_cluster = choice_cluster * allow_mask + (1 - allow_mask) * num_clusters

        # compute new centers
        center_shift = torch.zeros(1, dtype=torch.float32).to(device)
        new_states = []
        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)

            if selected.shape[0] > 0:
                new_states.append(selected.mean(dim=0))
                center_shift += (new_states[-1] - initial_state[index]).pow(2).sum().sqrt()
            else:
                new_states.append(initial_state[index])

        num_clusters = len(new_states)
        initial_state = torch.stack(new_states)

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        # tqdm_meter.set_postfix(
        #     iteration=f'{iteration}',
        #     center_shift=f'{center_shift.item() ** 2:0.6f}',
        #     tol=f'{tol:0.6f}'
        # )
        # tqdm_meter.update()
        if center_shift ** 2 < tol:
            break

    return choice_cluster, initial_state


def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1)
    return dis.sqrt()


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis