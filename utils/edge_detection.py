import cv2
import numpy as np
from skimage.morphology import skeletonize

from utils.image import image_histogram_equalization, set_image_height_without_distortion


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