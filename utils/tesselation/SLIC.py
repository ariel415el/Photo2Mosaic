import os

import cv2
import numpy as np
from skimage import img_as_float
from skimage.color import rgb2lab
from tqdm import tqdm

from utils.tesselation.common import sample_centers, smooth_centers_by_gradients
from utils import plot_label_map, overlay_rgb_edges, StructureTensor
from scipy.ndimage.measurements import label as scipy_label


def color_dist(points, center):
    return ((points - center) ** 2).sum(-1)


def spacial_dist(points, center, structure, mask):
    first_axis_factor = 1
    directions = None
    if mask is not None:
        directions = structure.get_structure_tensor(mask)

    if directions is None:
        first_axis_factor = 1
        directions = np.eye(2)

    diffs = points - center

    return np.power((diffs @ directions[0]), 2) * first_axis_factor + np.power((diffs @ directions[1]), 2)


def simplify_blobs(label_map, centers):
    """
    Convert the label of each blob to the label of its nearest center with onnected components
    """
    h, w = label_map.shape[:2]
    new_label_map = label_map.copy()
    yx_field = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=2)[..., ::-1]

    for i in range(len(centers)):
        blobs, ncomponents = scipy_label((label_map == i))
        blobs = [(blobs == i) for i in range(1, ncomponents + 1)]
        blobs = sorted(blobs, key=lambda x: x.sum())[::-1]
        for blob in blobs:
            blob_center = np.mean(yx_field[np.where(blob)], axis=0)
            dists = np.linalg.norm(centers - blob_center, axis=-1)
            blob_label = np.argmin(dists)
            new_label_map[np.where(blob)] = blob_label

    return new_label_map

def SLIC_superPixels(input_img, density_map, n_tiles, n_iters=10, search_area_factor=2, debug_dir=None):
    """
    Creates SLIC superpixels using spacial and color distances
    @param input_img: image to segment
    @param density_map: unsigned integer map. The higher the number the denser will the cells in that area be
    @param density_map: unsigned integer map. The higher the number the denser will the cells in that area be
    @param n_tiles: approximately How many cells will there be
    @param n_iters: This is an iterative algorithm
    @param search_area_factor: To reduce memory and compute the search around each cell is restricted to small sorounding
    @return:
    """

    structure = StructureTensor(cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY))
    label_map = None
    compactness = 10
    h, w = input_img.shape[:2]

    centers = sample_centers((h, w), n_tiles, 'uniform', density_map)
    centers = smooth_centers_by_gradients(centers, cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY))

    lab_field = rgb2lab(img_as_float(input_img)) / compactness
    yx_field = np.indices((h,w)).transpose(1,2,0)

    S = int(np.ceil(np.sqrt(h * w / n_tiles)))
    color_normalizers = np.ones(len(centers)) * compactness
    SA = S * search_area_factor
    min_dist_map = np.inf * np.ones((len(centers), h, w), np.float32)
    for iter in tqdm(range(n_iters)):
        for c_idx in range(len(centers)):
            cy, cx = centers[c_idx]
            search_slice = tuple([slice(max(0, cy - SA), cy + SA), slice(max(0, cx - SA), cx + SA)])

            color_distances = color_dist(lab_field[search_slice],  lab_field[cy, cx])
            spacial_distances = spacial_dist(yx_field[search_slice], yx_field[cy, cx], structure, label_map == c_idx)

            dist_map = spacial_distances / (S**2) + color_distances / (color_normalizers[c_idx])

            min_dist_map[c_idx][search_slice] = dist_map * density_map[cy, cx]

        label_map = np.argmin(min_dist_map, axis=0)
        label_map[np.where(np.min(min_dist_map, axis=0) == np.inf)] = -1

        # Recenter clusters and delete empty ones
        remove_indices = []
        for c_idx in range(len(centers)):
            if (label_map == c_idx).any():
                centers[c_idx] = np.round(np.mean(yx_field[label_map == c_idx], axis=0)).astype(int)
            else:
                remove_indices.append(c_idx)

        # Update removed centers
        centers = np.delete(centers, remove_indices, axis=0)
        color_normalizers = np.delete(color_normalizers, remove_indices, axis=0)
        if remove_indices:
            for c_idx in remove_indices:
                label_map[label_map >= c_idx] -=1

        # SLICO: Update each center's color normalization to maximum color distance
        for c_idx in range(len(centers)):
            cy, cx = centers[c_idx]
            max_color_dist = color_dist(lab_field[label_map == c_idx], lab_field[cy, cx]).max()
            color_normalizers[c_idx] = max(compactness, max_color_dist)

        if debug_dir:
            plot_label_map(label_map, centers, path=os.path.join(debug_dir, f'Clusters_{iter}.png'))

    label_map = simplify_blobs(label_map, centers)

    if debug_dir:
        plot_label_map(label_map, centers, path=os.path.join(debug_dir, f'Connected_Clusters.png'))
        overlay_rgb_edges(input_img, label_map, path=os.path.join(debug_dir, f'Overlayed_clusters.png'))

    return centers, label_map

def scikit_SLICO(input_img, debug_dir):
    from skimage.segmentation.slic_superpixels import slic
    label_map = slic(input_img, n_segments=2000, slic_zero=True)
    yx_field = np.indices(input_img.shape[:2]).transpose(1,2,0)
    centers = []
    for label in np.unique(label_map):
        centers += [np.mean(yx_field[label_map == label], axis=0).astype(int)]

    centers = np.array(centers)

    if debug_dir:
        plot_label_map(label_map, centers, path=os.path.join(debug_dir, f'Connected_Clusters.png'))
        overlay_rgb_edges(input_img, label_map, path=os.path.join(debug_dir, f'Overlayed_clusters.png'))

    return centers, label_map

