import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import ndimage
from tqdm import tqdm

from utils import plot_vornoi_cells, get_edge_map, render_tiles, get_rotation_matrix, plot_vector_field


def get_vornoi_cells(points, oritentations, direction_field):
    h, w = direction_field.shape[:2]
    coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1)

    gaps = coords - points[:, None, None]

    basis1 = (oritentations @ get_rotation_matrix(45))[:, None, None]
    basis2 = (oritentations @ get_rotation_matrix(-45))[:, None, None]

    with torch.no_grad():
        gaps = torch.from_numpy(gaps).cuda()
        basis1 = torch.from_numpy(basis1).cuda()
        basis2 = torch.from_numpy(basis2).cuda()

        distance_maps = torch.abs((gaps * basis1).sum(-1)) + torch.abs((gaps * basis2).sum(-1))

        distance_maps = distance_maps.cpu().numpy()

    vornoi_map = np.argmin(distance_maps, axis=0)

    centers = np.array([coords[vornoi_map==i].mean(0) for i in range(len(points))])

    # update orientations
    oritentations = direction_field[centers[:,0].astype(int), centers[:, 1].astype(int)]


    return vornoi_map, centers, oritentations


def create_direction_field(image):
    edge_map = get_edge_map(image)
    dist_transform = ndimage.distance_transform_edt(edge_map == 0)

    direction_field = np.stack(np.gradient(dist_transform), axis=2)

    direction_field = np.pad(direction_field, ((1, 1), (1, 1), (0, 0)), mode='reflect')


    norms = np.linalg.norm(direction_field, axis=2, keepdims=True)
    direction_field /= norms

    nans = norms[..., 0] == 0
    random_direction = np.random.rand(nans.sum(), 2)
    random_direction /= np.linalg.norm(random_direction, axis=1, keepdims=True)
    direction_field[nans] = random_direction

    return direction_field

def get_random_state(n_tiles, h, w):
    points = np.stack([np.random.random(n_tiles) * h, np.random.random(n_tiles) * w], axis=1)
    # oritentations = np.ones((n_tiles, 2))
    oritentations = np.random.rand(n_tiles, 2)
    oritentations /= np.linalg.norm(oritentations, axis=1, keepdims=True)

    return points, oritentations


def create_mosaic(reference_image):
    n_tiles = 1000
    delta = 0.75   # Direction field variation level
    h, w, _ = reference_image.shape
    alpha = 0.5
    tile_size = int(delta * np.sqrt(h*w/n_tiles))
    n_iters = 10

    direction_field = create_direction_field(reference_image)
    plot_vector_field(direction_field)

    points, oritentations = get_random_state(n_tiles, h, w)

    for i in tqdm(range(n_iters)):
        vornoi_map, points, oritentations = get_vornoi_cells(points, oritentations, direction_field)

    plot_vornoi_cells(vornoi_map, points, oritentations)

    render_tiles(points, oritentations, reference_image, tile_size)


if __name__ == '__main__':

    img = cv2.imread('/home/ariel/projects/Photo2Mosaic/YingYang.png')
    # img = cv2.resize(img, (512,512))
    create_mosaic(img)
