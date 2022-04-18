import os

import numpy as np
from tqdm import tqdm

import torch
from utils.tesselation.common import sample_centers, ROTATION_MATRICES, get_biggest_blob_mask
from utils import plot_label_map

def get_fixed_size_slice(center, interval_size, window_size):
    s = window_size // 2
    if center - s < 0:
        array_slice = slice(0, s)
    elif center > interval_size - s:
        array_slice = slice(interval_size - s, interval_size)
    else:
        array_slice = slice(center - s, center + s)
    return array_slice

def oriented_tesselation_with_edge_avoidance(edge_map, direction_map, density_map, n_tiles, n_iters=10, search_area_factor=2, cut_by_edges=False, debug_dir=None):
    """
    Creates a map of Vornoi Cells with oriented L1 distances
    @param edge_map: edges to avoid. avoidance is determined by cut_by_edges
    @param direction_map: a 2d vector specifying the desired orientation in each image pixel
    @param density_map: unsigned integer map. The higher the number the denser will the cells in that area be
    @param n_tiles: approximately How many cells will there be
    @param n_iters: This is an iterative algorithm
    @param search_area_factor: To reduce memory and compute the search around each cell is restricted to small sorounding
    @param cut_by_edges: False: avoid edges by simply ignoring edge pixels (Hausner 2001). True: Cut cells that overlap edges
    @param debug_dir:
    @return:
    """
    h, w = edge_map.shape[:2]

    centers = sample_centers((h, w), n_tiles, 'random', density_map)

    yx_field = np.indices((h,w)).transpose(1,2,0)

    S = int(np.ceil(np.sqrt(h * w / n_tiles))) * search_area_factor
    pbar = tqdm(range(n_iters))

    for iter in pbar:
        pbar.set_description(f"N-centers: {len(centers)}")
        min_dist_map = np.inf * np.ones((len(centers), h, w), np.float16)
        soroundings = []
        for c_idx in range(len(centers)):
            cy, cx = centers[c_idx]
            orientation = direction_map[cy, cx]
            basis1 = orientation @ ROTATION_MATRICES[-45]
            basis2 = orientation @ ROTATION_MATRICES[45]

            search_slice = tuple([get_fixed_size_slice(cy, h, S), get_fixed_size_slice(cx, h, S)])

            diffs = yx_field[search_slice] - yx_field[cy, cx]

            dist_map = np.abs((diffs @ basis1)) + np.abs((diffs @ basis2))
            dist_map = dist_map * density_map[cy, cx]

            if cut_by_edges and edge_map[search_slice].any():
                mask = get_biggest_blob_mask(edge_map[search_slice])
                min_dist_map[c_idx][search_slice][mask] = dist_map[mask]
            else:
                min_dist_map[c_idx][search_slice] = dist_map

        label_map = np.argmin(min_dist_map, axis=0)
        label_map[edge_map == 1] = -1

        remove_indices = []
        for c_idx in range(len(centers)):
            if (label_map == c_idx).any():
                centers[c_idx] = np.round(np.mean(yx_field[label_map == c_idx], axis=0)).astype(int)
            else:
                remove_indices.append(c_idx)

        centers = np.delete(centers, remove_indices, axis=0)
        if debug_dir:
            plot_label_map(label_map, centers, path=os.path.join(debug_dir, f'Clusters_{iter}.png'))
    if debug_dir:
        plot_label_map(label_map, centers, centers, path=os.path.join(debug_dir, f'Connected_Clusters.png'))

    return centers, label_map