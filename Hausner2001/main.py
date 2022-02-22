import os.path

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm

from utils import plot_vornoi_cells, render_tiles, get_rotation_matrix, plot_vector_field


def get_edges_map(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
    # mask = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    mask = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

    mask[mask==255] = 1

    return mask

def normalize_directions(vectors):
    # Normalize vector filed
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms

    # Replace zero vectors with random directions
    nans = norms[:, 0] == 0
    random_direction = np.random.rand(nans.sum(), 2)
    random_direction /= np.linalg.norm(random_direction, axis=1, keepdims=True)
    vectors[nans] = random_direction

    return vectors

def get_avoidance_map(edges_map):
    # edges_map = binary_dilation(edges_map, iterations=1).astype(np.uint8)
    # mask = binary_erosion(mask, iterations=1).astype(np.uint8)

    return edges_map


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

    for v in np.unique(alpha_mask):
        distance_maps[alpha_mask[points[:, 0], points[:, 1]] == v] *= v

    vornoi_map = np.argmin(distance_maps, axis=0)


    # Mark edges as ex-teritory (push cells away from edges)
    vornoi_map[avoidance_map == 1] = vornoi_map.max() + 2

    # ensure non-empty cells (points may not be closest to itself since argmax between same values is arbitraray)
    vornoi_map[points[:, 0], points[:, 1]] = np.arange(len(points))

    # Move points to be the centers of their according vornoi cells
    centers = np.array([coords[vornoi_map==i].mean(0) for i in range(len(points))]).astype(np.uint64)

    # update orientations using vector field
    oritentations = direction_field[centers[:,0].astype(int), centers[:, 1].astype(int)]
    oritentations = normalize_directions(oritentations)

    return vornoi_map, centers, oritentations


def create_direction_field(edge_map):
    """
    Create a direction vector field from edges in the edges_map.
    Compute edges map its distance transform and then its gradient map.
    Normalize gradient vector to have unit norm.
    """

    dist_transform = ndimage.distance_transform_edt(edge_map == 0)

    direction_field = np.stack(np.gradient(dist_transform), axis=2)

    return direction_field


def get_initial_state(n_tiles, h, w):
    """Sample random vornoi cell centers and directions"""
    # points = np.stack([np.random.random(n_tiles) * h, np.random.random(n_tiles) * w], axis=1).astype(np.uint64)
    s = int(np.ceil(np.sqrt(n_tiles)))
    y, x = np.meshgrid(np.linspace(1, h - 1, s), np.linspace(1, w - 1, s))
    points = np.stack([y.flatten(), x.flatten()], axis=1).astype(np.uint64)[:n_tiles]
    # oritentations = np.ones((n_tiles, 2))
    oritentations = np.random.rand(n_tiles, 2)
    oritentations /= np.linalg.norm(oritentations, axis=1, keepdims=True)

    return points, oritentations


def create_mosaic(reference_image, outputs_dir):
    os.makedirs(outputs_dir, exist_ok=True)

    h, w, _ = reference_image.shape
    tile_size = int(delta * np.sqrt(h * w / n_tiles))

    edge_map = get_edges_map(reference_image)

    avoidance_map = get_avoidance_map(edge_map)

    direction_field = create_direction_field(edge_map)

    plot_vector_field(reference_image, direction_field, path=os.path.join(outputs_dir, 'VectorField.png'))

    from matplotlib import pyplot as plt
    plt.imshow(cv2.cvtColor(avoidance_map * 255, cv2.COLOR_GRAY2RGB))
    plt.savefig(os.path.join(outputs_dir, 'EdgeMap.png'))
    plt.clf()

    points, oritentations = get_initial_state(n_tiles, h, w)

    losses = []
    for i in tqdm(range(n_iters)):
        vornoi_map, points, oritentations = get_vornoi_cells(points, oritentations, direction_field, avoidance_map)
        plot_vornoi_cells(vornoi_map, points, oritentations, avoidance_map, path=os.path.join(outputs_dir, f'Vornoi_diagram_{i}.png'))
        loss = render_tiles(points, oritentations, reference_image, tile_size, alpha_mask, path=os.path.join(outputs_dir, f'Mosaic_{i}.png'))
        losses.append(loss)

        plt.plot(np.arange(len(losses)), losses)
        plt.savefig(os.path.join(outputs_dir, 'Loss.png'))
        plt.clf()


if __name__ == '__main__':
    img_path = '../images/YingYang.png'
    img_path = '../images/Elon.jpg'
    # img_path = '../images/Elat1.jpg'
    # img_path = '../images/diagonal.png'

    mask_path = '../images/Elon_mask.jpg'

    device = torch.device("cpu")
    resize = 256
    n_tiles = 2000
    delta = 0.75  # Direction field variation level
    n_iters = 20

    im_name = os.path.basename(os.path.splitext(img_path)[0])
    outputs_dir = os.path.join("outputs", f"{im_name}_R-{resize}_N-{n_tiles}_dlta-{delta}")

    img = cv2.imread(img_path)
    img = cv2.resize(img, (int(resize * img.shape[1] / img.shape[0]), resize))

    alpa_mask = cv2.imread(mask_path)
    alpa_mask = cv2.resize(alpa_mask, (int(resize * alpa_mask.shape[1] / alpa_mask.shape[0]), resize))
    alpha_mask = cv2.cvtColor(alpa_mask, cv2.COLOR_BGR2GRAY)
    alpha_mask[alpha_mask <= 127] = 1
    alpha_mask[alpha_mask > 127] = 2
    create_mosaic(img, outputs_dir)
