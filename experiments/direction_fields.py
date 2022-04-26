import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_closing, binary_opening, binary_erosion
import lic
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import read_edges_map, create_direction_field, plot_vector_field, get_rotation_matrix, edge_detection, \
    normalize_vector_field, line_interval_convolution, iterative_edge_tangent_flow

def create_direction_field(edge_map):
    """
    Create a direction vector field from edges in the edges_map.
    Compute edges map its distance transform and then its gradient map.
    Normalize gradient vector to have unit norm.
    """
    dist_transform = ndimage.distance_transform_edt(edge_map == 0)
    dist_transform = cv2.GaussianBlur(dist_transform, (5, 5), 0)
    direction_field = np.stack(np.gradient(dist_transform), axis=2)
    return direction_field

def get_dist_transform_tangent_gradient(edge_map):
    direction_field = create_direction_field(edge_map)
    direction_field = direction_field @ get_rotation_matrix(90)
    return direction_field

def get_sobel_tangent_and_magnitude(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    sobelmag = np.sqrt(np.multiply(sobelx, sobelx) + np.multiply(sobely, sobely))
    sobelmag = np.divide(sobelmag, np.amax(sobelmag))

    tangent = np.stack([-1 * sobely, sobelx], axis=2)
    tangnorm = np.linalg.norm(tangent, axis=2)
    np.place(tangnorm, tangnorm == 0, [1])
    tangent /= tangnorm[...,None]

    return tangent, sobelmag


def color_angle(direction_field, path=None):
    radians = np.arctan2(direction_field[...,0], direction_field[...,1])
    plt.imshow(radians, cmap='hot')
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()


if __name__ == '__main__':
    image = cv2.imread('/home/ariel/projects/Photo2Mosaic/Photo2Mosaic/images/images/beachBall.jpg')
    # image = cv2.resize(image, (256,256))

    # Get edge map
    edge_map = read_edges_map('/home/ariel/projects/Photo2Mosaic/Photo2Mosaic/images/edge_maps/turk_edges.png', t=20)
    # edge_map = edge_detection.get_edges_map_canny(image)
    # sobel_tangent, sobel_mag = get_sobel_tangent_and_magnitude(image)

    direction_field = get_dist_transform_tangent_gradient(edge_map)

    direction_field = cv2.GaussianBlur(direction_field, (5, 5), 0)

    # direction_field = iterative_edge_tangent_flow(direction_field, edge_map, iterations=10, mu=7)

    line_interval_convolution(direction_field)
    color_angle(direction_field)
