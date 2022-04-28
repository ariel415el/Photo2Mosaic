import os

import cv2
import numpy as np
from skimage.morphology import skeletonize
from tqdm import tqdm

from utils.image import image_histogram_equalization, set_image_height_without_distortion


def get_edges_map(mode, image, mask=None):
    """Get an edge map in 3 ways:
     1: From the image itself using an edge detector
     2: Use the contour of the mask (density_map) as edges
     3: Use a predefined edge image. Possibly refine it to its skeleton
     """
    if mode == 'mask' and mask is not None:
        edge_map = get_edges_map_canny(cv2.cvtColor(mask * 127, cv2.COLOR_GRAY2BGR))
    elif os.path.exists(mode):
        edge_map = read_edges_map(mode, t=127, resize=image.shape[0])
    else:
        edge_map = get_edges_map_canny(image, blur_size=7, sigma=5, t1=100, t2=150)

    return edge_map


def get_edges_map_DiBlasi2005(image):
    """
    Compute edges as described in the paper
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_gray = image_histogram_equalization(img_gray)

    # Blur the image for better edge detection
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    mean, std = img_gray.mean(), img_gray.std()
    T = std / 4

    img_gray[np.abs(img_gray - mean) > T] = 1
    img_gray[np.abs(img_gray - mean) <= T] = 0

    edges_map = np.absolute(cv2.Laplacian(img_gray, cv2.CV_64F)).astype(np.uint8)
    # edges_map = binary_opening(edges_map, iterations=1).astype(np.uint8)

    return edges_map


def get_edges_map_canny(image, blur_size=7, sigma=None, t1=50, t2=100):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if sigma is None:
        sigma = int(0.01 * np.mean(image.shape[:2]))

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (blur_size, blur_size), sigma)

    # mask = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    mask = cv2.Canny(image=img_blur, threshold1=t1, threshold2=t2)

    mask[mask==255] = 1

    return mask


def read_edges_map(edges_map_path, t=127, resize=None):
    edge_map = cv2.imread(edges_map_path, cv2.IMREAD_GRAYSCALE)
    if resize:
        edge_map = set_image_height_without_distortion(edge_map, resize, mode=cv2.INTER_NEAREST)
    edge_map[edge_map < t] = 0
    edge_map[edge_map >= t] = 1
    edge_map = skeletonize(edge_map).astype(np.uint8)
    return edge_map


def iterative_edge_tangent_flow(direction_field, magnitude_map, mu=5, iterations=3):
    """
    According to the paper "Imaging Vector Fields Using Line Integral Convolution"
    Smooth a vector field with a weighted bilateral filter
    """

    h,w = direction_field.shape[:2]

    pbar = tqdm(total=h*w*iterations)
    for iteration in range(iterations):
        new_direction_field = np.zeros((h, w, 2))
        for r in range(h):
            for c in range(w):

                h_slice = slice(max(0, r - mu), r + mu)
                w_slice = slice(max(0, c - mu), c + mu)
                mag_weights = (magnitude_map[r,c] - magnitude_map[h_slice, w_slice] + 1) / 2
                direction_weights = direction_field[h_slice, w_slice] @ direction_field[r,c]
                weights = mag_weights * direction_weights
                weights[direction_weights < 0] *= -1

                new_direction_field[r][c] = np.sum(direction_field[h_slice, w_slice] * weights[..., None], axis=(0,1))
                weight_sum = np.sum(weights[..., None], axis=(0,1))
                if weight_sum > 0 :
                    new_direction_field[r][c] /= weight_sum

                pbar.update(1)

        direction_field = new_direction_field

    return direction_field
