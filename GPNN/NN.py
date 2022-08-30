import torch


def get_NN_indices(X, Y, alpha, b=128, metric='l2'):
    """
    Get the nearest neighbor index from Y for each X. Use batches to save memory
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """
    dist = compute_distances_matrix_in_batches(X, Y, b, metric)
    # dist = torch.cdist(X.view(len(X), -1), Y.view(len(Y), -1)) # Not enough memory
    dist = (dist / (torch.min(dist, dim=0)[0] + alpha)) # compute_normalized_scores
    NNs = torch.argmin(dist, dim=1)  # find_NNs
    return NNs

def compute_distances_matrix_in_batches(X, Y, b, metric):
    """
    Computes distance matrix in batches of rows to reduce memory consumption from (n1 * n2 * d) to (d * n2 * d)
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    :param b: rows batch size
    Returns a (n2, n1) matrix of L2 distances
    """
    """"""
    b = min(b, len(X))
    dist_mat = torch.zeros((X.shape[0], Y.shape[0]), dtype=torch.float16, device=X.device)
    n_batches = len(X) // b
    for i in range(n_batches):
        dist_mat[i * b:(i + 1) * b] = efficient_compute_distances_matrix(X[i * b:(i + 1) * b], Y, metric)
    if len(X) % b != 0:
        dist_mat[n_batches * b:] = efficient_compute_distances_matrix(X[n_batches * b:], Y, metric)

    return dist_mat


def efficient_compute_distances_matrix(X, Y, metric):
    """
    Pytorch efficient way of computing distances between all vectors in X and Y, i.e (X[:, None] - Y[None, :])**2
    Get the nearest neighbor index from Y for each X
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """
    if metric == 'cosine':
        similarities = torch.mm(X, torch.transpose(Y, 0, 1))
        similarities /= (torch.norm(X, dim=1)[:, None] * torch.norm(Y, dim=1)[None, ])
        dist = 1 - similarities
    elif metric == 'l2':
        dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
    else:
        raise ValueError("No such metric")
    d = X.shape[1]
    dist /= d # normalize by size of vector to make dists independent of the size of d ( use same alpha for all patche-sizes)
    return dist


def get_col_mins_efficient(X, Y, b, metric):
    """
    Computes the l2 distance to the closest x or each y.
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns n1 long array of L2 distances
    """
    mins = torch.zeros(Y.shape[0], dtype=X.dtype, device=X.device)
    n_batches = len(Y) // b
    for i in range(n_batches):
        mins[i * b:(i + 1) * b] = efficient_compute_distances_matrix(X, Y[i * b:(i + 1) * b], metric).min(0)[0]
    if len(Y) % b != 0:
        mins[n_batches * b:] = efficient_compute_distances_matrix(X, Y[n_batches * b:], metric).min(0)[0]

    return mins

def get_NN_indices_low_memory(X, Y, alpha, b, metric='l2'):
    """
    Get the nearest neighbor index from Y for each X.
    Avoids holding a (n1 * n2) amtrix in order to reducing memory footprint to (b * max(n1,n2)).
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """
    if alpha is not None:
        normalizing_row = get_col_mins_efficient(X, Y, b, metric)
        normalizing_row = alpha + normalizing_row[None, :]
    else:
        normalizing_row = 1

    NNs = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    n_batches = len(X) // b
    for i in range(n_batches):
        dists = efficient_compute_distances_matrix(X[i * b:(i + 1) * b], Y, metric) / normalizing_row
        NNs[i * b:(i + 1) * b] = dists.min(1)[1]
    if len(X) % b != 0:
        dists = efficient_compute_distances_matrix(X[n_batches * b:], Y, metric) / normalizing_row
        NNs[n_batches * b:] = dists.min(1)[1]
    return NNs

