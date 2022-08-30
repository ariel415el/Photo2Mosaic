import cv2
import numpy as np
from scipy.ndimage import binary_erosion

from utils import get_rotation_matrix, simplify_contour
from scipy.ndimage.measurements import label as scipy_label

ROTATION_MATRICES={x: get_rotation_matrix(x) for x in [45, 90, -45]}

def sample_uniformly_from_area(mask, n, sparse=True):
    posible_points = np.stack(np.where(mask), axis=1)
    if sparse:
        density = mask.astype(float) / mask.sum()
        points = []
        for i in range(n):
            index = np.random.choice(len(posible_points), p=density[mask])
            point = posible_points[index]
            if len(points) > 0:
                assert not np.any(np.all(np.array(points) == np.array(point), axis=1))
            points.append(point)
            density[point[0], point[1]] = 0
            density /= density.sum()
        return np.array(points)
    else:
        indices = np.random.choice(len(posible_points), n, replace=False)
        return posible_points[indices]

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

                all_points += [sample_uniformly_from_area(density_map == i, num_dense_area_points, sparse=True)]

        default_positions = sample_uniformly_from_area(density_map == 1, num_default_area_points, sparse=True)
        centers = np.concatenate([default_positions] + all_points, axis=0).astype(int)

    else:  # uniform
        S = int(np.round(np.sqrt(h * w / N)))
        y, x = np.meshgrid(np.arange(S // 2, h, S), np.arange(S // 2, w, S))
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
        if edge_map[y,x] == 1:
            while radius < 3:
                neighberhood = edge_map[y-radius :y + radius + 1, x - radius:x + radius + 1]
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


def simplify_label_map(label_map, approximation_params, erode_blobs=False):
    if approximation_params[0] is None:
        return label_map
    new_label_map = -1 * np.ones_like(label_map)
    for label in np.unique(label_map):
        if label < 0:
            continue
        blob = (label_map == label)
        if erode_blobs:
            blob = binary_erosion(blob, iterations=1)

        contours, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if approximation_params[0] == "poly":
                cnt = cv2.approxPolyDP(cnt, approximation_params[1] * cv2.arcLength(cnt, True), True)
            elif approximation_params[0] == 'square':
                cnt = np.int0(cv2.boxPoints(cv2.minAreaRect(np.array(cnt).astype(int))))
            elif approximation_params[0] == 'fourier':
                cnt = simplify_contour(cnt, approximation_params[1])

            blob_map = np.zeros_like(label_map, dtype=np.uint8)
            cv2.drawContours(blob_map, [cnt], -1, color=1, thickness=cv2.FILLED)
            if approximation_params[0] == 'contour':
                cv2.drawContours(blob_map, [cnt], 0, color=-1, thickness=1)
            new_label_map[blob_map == 1] = label

    return new_label_map


def render_label_map_with_image(label_map, image, centers, output_path="mosaic.png"):
    canvas = np.ones_like(image) * 127
    n_edges_centers = 0
    for i in range(len(centers)):
        label = label_map[centers[i][0], centers[i][1]]
        n_edges_centers + 1
        color = image[centers[i][0], centers[i][1]]
        canvas[label_map == label] = color
    print(f"Got {n_edges_centers} edge centers")
    cv2.imwrite(output_path, canvas)

