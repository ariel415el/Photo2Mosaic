import os.path

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm

from utils import plot_vornoi_cells, get_edge_map, render_tiles, get_rotation_matrix, plot_vector_field


def get_vornoi_cells(points, oritentations, direction_field, avoidance_map):
    """
    Iterative approach for computing centridal vornoi cells using brute force equivalet of the Z-buffer algorithm
    For point in "points" compute a distance map now argmin over all maps to get a single map with index of
    closest point to each coordinate.
    The metric is used is the L1 distance with axis rotated by the points' orientation
    """

    h, w = direction_field.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([yy, xx], axis=-1)  # coords[a ,b] = [b,a]

    diffs = coords - points[:, None, None]

    basis1 = (oritentations @ get_rotation_matrix(45))[:, None, None]
    basis2 = (oritentations @ get_rotation_matrix(-45))[:, None, None]

    # Heavy computation on GPU
    with torch.no_grad():
        gaps = torch.from_numpy(diffs).to(device)
        basis1 = torch.from_numpy(basis1).to(device)  # u
        basis2 = torch.from_numpy(basis2).to(device)  # v

        # L1_u-v((x,y),(x0,y0)) = |<(x,y),u> - <(x0,y0),u>| + |<(x,y),v> - <(x0,y0),v>| = |<(x,y)-(x0,y0),u>| + |<(x,y)-(x0,y0),u>|
        distance_maps = torch.abs((gaps * basis1).sum(-1)) + torch.abs((gaps * basis2).sum(-1))
        distance_maps = distance_maps.cpu().numpy()

    vornoi_map = np.argmin(distance_maps, axis=0)

    # Mark edges as ex-teritory (push cells away from edges)
    vornoi_map[avoidance_map == 1] = vornoi_map.max() + 2

    # ensure non-empty cells (points may not be closest to itself since argmax between same values is arbitraray)
    vornoi_map[points[:, 0], points[:, 1]] = np.arange(len(points))

    # Move points to be the centers of their according vornoi cells
    centers = np.array([coords[vornoi_map==i].mean(0) for i in range(len(points))]).astype(np.uint64)

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

def get_avoidance_map(image):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (9, 9), 0)
    # mask = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    mask = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    mask[mask==255] = 1

    mask = binary_dilation(mask, iterations=1).astype(np.uint8)
    # mask = binary_erosion(mask, iterations=1).astype(np.uint8)

    return mask

def get_random_state(n_tiles, h, w):
    """Sample random vornoi cell centers and directions"""
    points = np.stack([np.random.random(n_tiles) * h, np.random.random(n_tiles) * w], axis=1).astype(np.uint64)
    # oritentations = np.ones((n_tiles, 2))
    oritentations = np.random.rand(n_tiles, 2)
    oritentations /= np.linalg.norm(oritentations, axis=1, keepdims=True)

    return points, oritentations


def create_mosaic(reference_image, outputs_dir):
    os.makedirs(outputs_dir, exist_ok=True)

    h, w, _ = reference_image.shape
    tile_size = int(delta * np.sqrt(h * w / n_tiles))

    direction_field = create_direction_field(reference_image)

    avoidance_map = get_avoidance_map(reference_image)

    plot_vector_field(reference_image, direction_field, path=os.path.join(outputs_dir, 'VectorField.png'))
    from matplotlib import pyplot as plt
    plt.imshow(cv2.cvtColor(avoidance_map * 255, cv2.COLOR_GRAY2RGB))
    plt.savefig(os.path.join(outputs_dir, 'EdgeMap.png'))
    plt.clf()

    points, oritentations = get_random_state(n_tiles, h, w)

    for i in tqdm(range(n_iters)):
        vornoi_map, points, oritentations = get_vornoi_cells(points, oritentations, direction_field, avoidance_map)
        plot_vornoi_cells(vornoi_map, points, oritentations, avoidance_map, path=os.path.join(outputs_dir, f'Vornoi_diagram_{i}.png'))

    render_tiles(points, oritentations, reference_image, tile_size, path=os.path.join(outputs_dir, 'Mosaic.png'))


if __name__ == '__main__':
    # img_path = '../images/YingYang.png'
    # img_path = '../images/Elon.jpg'
    img_path = '../images/Elat1.jpg'
    # img_path = '../images/diagonal.png'
    device = torch.device("cpu")
    resize = 512
    n_tiles = 1000
    delta = 0.7  # Direction field variation level
    n_iters = 20

    im_name = os.path.basename(os.path.splitext(img_path)[0])
    outputs_dir = os.path.join("outputs", f"{im_name}_R-{resize}_N-{n_tiles}_dlta-{delta}")

    img = cv2.imread(img_path)
    img = cv2.resize(img, (int(resize * img.shape[1] / img.shape[0]), resize))
    create_mosaic(img, outputs_dir)
