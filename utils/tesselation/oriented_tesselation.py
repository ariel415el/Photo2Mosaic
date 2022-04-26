import os

import cv2
import numpy as np
from tqdm import tqdm

import torch
from utils.tesselation.common import sample_centers, ROTATION_MATRICES, get_biggest_blob_mask
from utils import plot_label_map, get_rotation_matrix


def get_fixed_size_slice(center, interval_size, window_size):
    s = window_size // 2
    if center - s < 0:
        array_slice = slice(0, s)
    elif center > interval_size - s:
        array_slice = slice(interval_size - s, interval_size)
    else:
        array_slice = slice(center - s, center + s)
    return array_slice


def restrict_cell_shapes(label_map, centers, aprrox=False):
    new_label_map = -1 * np.ones_like(label_map).astype(np.uint16)
    for label in range(len(centers)):
        blob = (label_map == label)
        from scipy.ndimage import binary_erosion
        blob = binary_erosion(blob, iterations=1)

        contours, hierarchy = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if aprrox:
                shape = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(contours[0], True), True)
            else:
                shape = np.int0(cv2.boxPoints(cv2.minAreaRect(np.array(cnt).astype(int))))
            cv2.drawContours(new_label_map, [shape], -1, color=label, thickness=cv2.FILLED)

    return new_label_map

ROTATION_MATRICES[135] = get_rotation_matrix(135)
ROTATION_MATRICES[225] = get_rotation_matrix(225)
ROTATION_MATRICES[315] = get_rotation_matrix(315)
def replace_cells_with_squares(label_map, centers, direction_map, density_map, tile_size):
    new_label_map = -1 * np.ones_like(label_map).astype(np.uint16)

    oritentations = direction_map[centers[:, 0], centers[:, 1]]

    corner_directions = (oritentations @ np.stack([ROTATION_MATRICES[45],
                                                   ROTATION_MATRICES[135],
                                                   ROTATION_MATRICES[225],
                                                   ROTATION_MATRICES[315]]))

    corner_diameter = tile_size / np.sqrt(2)
    for i in range(centers.shape[0]):
        alpha = density_map[centers[i][0], centers[i][1]]

        contours = centers[i] + corner_directions[:, i] * corner_diameter / alpha
        contours = contours[:, ::-1]

        shape = np.int0(cv2.boxPoints(cv2.minAreaRect(np.array(contours).astype(int))))
        cv2.drawContours(new_label_map, [shape], -1, color=i, thickness=cv2.FILLED)

    return new_label_map

def oriented_tesselation_with_edge_avoidance(edge_map, direction_map, density_map, n_tiles, n_iters=10, search_area_factor=2, cut_by_edges=False, debug_dir=None):
    """
    Creates a map of Vornoi Cells with oriented L1 distances considering an edge map
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
    # orientations = direction_map[centers[:, 0], centers[:, 1]]

    yx_field = np.indices((h,w)).transpose(1,2,0)

    S = int(np.ceil(np.sqrt(h * w / n_tiles))) * search_area_factor

    pbar = tqdm(range(n_iters))
    for iter in pbar:
        pbar.set_description(f"N-centers: {len(centers)}")
        min_dist_map = np.inf * np.ones((len(centers), h, w), np.float16)
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

        label_map = restrict_cell_shapes(label_map, centers)
        # label_map = replace_cells_with_squares(label_map, centers, direction_map, density_map, tile_size=S//search_area_factor)

        remove_indices = []
        for c_idx in range(len(centers)):
            if (label_map == c_idx).any():
                centers[c_idx] = np.round(np.mean(yx_field[label_map == c_idx], axis=0)).astype(int)
            else:
                remove_indices.append(c_idx)

        centers = np.delete(centers, remove_indices, axis=0)

        # # Adjust orientations according to neihbors
        # orientations = direction_map[centers[:, 0], centers[:, 1]]
        # from scipy.spatial import distance_matrix
        # dist_matrix = distance_matrix(centers, centers)
        # closest_centers = np.argsort(dist_matrix, axis=1)
        # for c_idx in range(len(centers)):
        #     neighbors_orientation = orientations[closest_centers[c_idx][1:4]].mean(0)
        #     orientations[c_idx] = orientations[c_idx] * 0.5 + neighbors_orientation*0.5

        if debug_dir:
            plot_label_map(label_map, centers, path=os.path.join(debug_dir, f'Clusters_{iter}.png'))
    if debug_dir:
        plot_label_map(label_map, centers, centers, path=os.path.join(debug_dir, f'Connected_Clusters.png'))

    return centers, label_map