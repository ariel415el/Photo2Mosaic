import os.path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import ndimage
from tqdm import tqdm

from utils import plot_vornoi_cells, get_edge_map, render_tiles, get_rotation_matrix, plot_vector_field


def get_vornoi_cells(points, oritentations, direction_field):
    """
    Iterative approach for computing centridal vornoi cells using brute force equivalet of the Z-buffer algorithm
    For point in "points" compute a distance map now argmin over all maps to get a single map with index of
    closest point to each coordinate.
    The metric is used is the L1 distance with axis rotated by the points' orientation
    """

    h, w = direction_field.shape[:2]
    coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1)  # coords[a ,b] = [b,a]


    diffs = coords - points[:, None, None]

    basis1 = (oritentations @ get_rotation_matrix(45))[:, None, None]
    basis2 = (oritentations @ get_rotation_matrix(-45))[:, None, None]

    # Heavy computation on GPU
    with torch.no_grad():
        gaps = torch.from_numpy(diffs).cuda()
        basis1 = torch.from_numpy(basis1).cuda()  # u
        basis2 = torch.from_numpy(basis2).cuda()  # v

        # L1_u-v((x,y),(x0,y0)) = |<(x,y),u> - <(x0,y0),u>| + |<(x,y),v> - <(x0,y0),v>| = |<(x,y)-(x0,y0),u>| + |<(x,y)-(x0,y0),u>|
        distance_maps = torch.abs((gaps * basis1).sum(-1)) + torch.abs((gaps * basis2).sum(-1))
        distance_maps = distance_maps.cpu().numpy()

    vornoi_map = np.argmin(distance_maps, axis=0)

    # Move points to be the centers of their according vornoi cells
    centers = np.array([coords[vornoi_map==i].mean(0) for i in range(len(points))])

    # update orientations using vector field
    oritentations = direction_field[centers[:,0].astype(int), centers[:, 1].astype(int)]

    return vornoi_map, centers, oritentations


def create_direction_field(image):
    """
    Create a direction vector field from edges in the image.
    Compute edges map its distance transform and then its gradient map.
    Normalize gradient vector to have unit norm.
    """

    edge_map = get_edge_map(image)
    dist_transform = ndimage.distance_transform_edt(edge_map == 0)

    direction_field = np.stack(np.gradient(dist_transform), axis=2)

    # Normalize vector filed
    norms = np.linalg.norm(direction_field, axis=2, keepdims=True)
    direction_field /= norms

    # Replace zero vectors with random directions
    nans = norms[..., 0] == 0
    random_direction = np.random.rand(nans.sum(), 2)
    random_direction /= np.linalg.norm(random_direction, axis=1, keepdims=True)
    direction_field[nans] = random_direction

    return direction_field

def get_random_state(n_tiles, h, w):
    """Sample random vornoi cell centers and directions"""
    points = np.stack([np.random.random(n_tiles) * h, np.random.random(n_tiles) * w], axis=1)
    # oritentations = np.ones((n_tiles, 2))
    oritentations = np.random.rand(n_tiles, 2)
    oritentations /= np.linalg.norm(oritentations, axis=1, keepdims=True)

    return points, oritentations


def create_mosaic(reference_image, outputs_dir):
    h, w, _ = reference_image.shape
    tile_size = int(delta * np.sqrt(h * w / n_tiles))
    os.makedirs(outputs_dir, exist_ok=True)
    direction_field = create_direction_field(reference_image)
    plot_vector_field(direction_field, path=os.path.join(outputs_dir, 'VectorField.png'))

    points, oritentations = get_random_state(n_tiles, h, w)

    for i in tqdm(range(n_iters)):
        vornoi_map, points, oritentations = get_vornoi_cells(points, oritentations, direction_field)
        plot_vornoi_cells(vornoi_map, points, oritentations, path=os.path.join(outputs_dir, f'Vornoi_diagram_{i}.png'))

    render_tiles(points, oritentations, reference_image, tile_size, path=os.path.join(outputs_dir, 'Mosaic.png'))


if __name__ == '__main__':
    img_path = '../images/YingYang.png'
    # img = cv2.resize(img, (512,512))
    alpha = 0.5
    n_tiles = 1000
    delta = 0.75  # Direction field variation level
    n_iters = 20

    im_name = os.path.basename(os.path.splitext(img_path)[0])
    outputs_dir = f"{im_name}_N-{n_tiles}_It-{n_iters}_dlta-{delta}"

    img = cv2.imread(img_path)
    create_mosaic(img, outputs_dir)
