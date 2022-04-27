import numpy as np
from utils import get_rotation_matrix
from scipy.ndimage.measurements import label as scipy_label

ROTATION_MATRICES={x: get_rotation_matrix(x) for x in [45, 90, -45]}

def sample_centers(image_shape, N, init_mode, density_map=None):
    """Sample initial tile centers and orientations  cell centers and directions"""
    h, w = image_shape
    if init_mode == 'random' and density_map is not None:  # sample randomly in each are in proportion to tile sizes in it
        all_points = []
        num_default_area_points = N
        for i in np.unique(density_map):
            if i != 1:
                dense_area_proportion = (density_map == i).sum() / density_map.size

                # if X is the num_dense_area_points then the sparse area has X/4 and solving leads to
                num_dense_area_points = int(4 * dense_area_proportion * N / (1 + 4 * dense_area_proportion))
                num_default_area_points -= num_dense_area_points

                posible_points = np.stack(np.where(density_map == i), axis=1)
                dense_positions = posible_points[np.random.randint(len(posible_points), size=num_dense_area_points)]
                all_points += [dense_positions]

        posible_points = np.stack(np.where(density_map == 1), axis=1)
        default_positions = posible_points[np.random.randint(len(posible_points), size=num_default_area_points)]
        centers = np.concatenate([default_positions] + all_points, axis=0).astype(int)

    else:  # uniform
        S = int(np.ceil(np.sqrt(h * w / N)))
        y, x = np.meshgrid(np.arange(S // 2 - 1, h - S // 2 + 1, S), np.arange(S // 2 - 1, w - S // 2 + 1, S))
        centers = np.stack([y.flatten(), x.flatten()], axis=1)

    return centers


def sample_random_normals(N):
    oritentations = np.random.rand(N, 2)
    oritentations /= np.linalg.norm(oritentations, axis=1, keepdims=True)
    return oritentations


def smooth_centers_by_gradients(centers, gray_image):
    """
    Move cluster centers to the lowest gradient position in a 3x3 neighborhood.
    """
    gradient_magnitudes = np.linalg.norm(np.stack(np.gradient(gray_image), axis=-1), axis=-1)
    gradient_magnitudes = np.pad(gradient_magnitudes, 1, 'constant', constant_values=np.inf)
    for i in range(len(centers)):
        r, c = centers[i]
        neighberhood = gradient_magnitudes[r :r + 3, c:c + 3] # r -> r+1 because of padding
        argmin = np.where(neighberhood == np.amin(neighberhood))
        centers[i] = (r + argmin[0][0] - 1, c + argmin[1][0] -1)  # -1 for offset from center
    return centers

def smooth_centers_by_edges(centers, edge_map):
    """
    Avoid centers from being placed on an edge
    """
    for i in range(len(centers)):
        y, x = centers[i]
        radius = 1
        if edge_map[y,x] != 1:
            continue
        while radius < 3:
            neighberhood = edge_map[y-radius :y + radius + 1, x - radius:x + radius + 1]
            if 1 in neighberhood:
                nwhere = np.where(neighberhood == 1)
                centers[i] = nwhere[0][0], nwhere[0][1]
                break
            radius += 1

    return centers

def get_contigous_blob_mask(edge_map, center=None):
    """If edge map (ususly a slice of it is the input) is split by edges then find the biggest/center-containing connected component mask"""
    all_blobs, ncomponents = scipy_label(1 - edge_map)
    if center is None:
        return all_blobs == np.argmax([(all_blobs == i).sum() for i in np.unique(all_blobs)])
    else:
        return all_blobs == all_blobs[center[0], center[1]]

