import cv2
import numpy as np
from scipy import ndimage

from utils.edge_detection import iterative_edge_tangent_flow


def create_direction_field(edge_map, smooth_directions_with_bilateral_filter=False):
    """
    Create a direction vector field from edges in the edges_map.
    Compute edges map its distance transform and then its gradient map.
    Normalize gradient vector to have unit norm.
    """
    dist_transform = ndimage.distance_transform_edt(edge_map == 0)
    dist_transform = cv2.GaussianBlur(dist_transform, (5, 5), 0)
    direction_field = np.stack(np.gradient(dist_transform), axis=2)

    if smooth_directions_with_bilateral_filter:
        direction_field = iterative_edge_tangent_flow(direction_field, edge_map)
    else:
        direction_field = cv2.GaussianBlur(direction_field, (5, 5), 0)

    direction_field = normalize_vector_field(direction_field)

    return direction_field, dist_transform


def get_level_lines(edges_map, density_map, tile_size, offset=0):
    if offset == 0:
        level_matrix = edges_map.copy()
    else:
        level_matrix = np.zeros_like(edges_map)

    direction_field, dist_transform = create_direction_field(edges_map)
    dist_map = dist_transform.astype(int)

    for factor in np.unique(density_map):
        gap_size = tile_size // factor
        mask = np.remainder(dist_map, gap_size) == int(offset * gap_size)
        mask = mask & (density_map == factor)
        level_matrix[mask] = 1

    return level_matrix, direction_field, dist_transform


def normalize_vector_field(vector_field):
    """
    Divide an np array of shape (H,W,d) by its norm in the last axis, replace zero divisions by random vectors
    """
    d = vector_field.shape[-1]
    norms = np.linalg.norm(vector_field, axis=-1)
    vector_field /= norms[..., None]
    if 0 in norms:
        nans = np.where(norms == 0)
        random_direction = np.random.rand(len(nans[0]), d)
        random_direction /= np.linalg.norm(random_direction, axis=1, keepdims=True)
        vector_field[nans] = random_direction

    return vector_field
