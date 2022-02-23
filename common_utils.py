import cv2
import numpy as np
from matplotlib import pyplot as plt

from skimage import color
from scipy.ndimage.morphology import binary_dilation


def plot_vector_field(vector_field, image=None, path="vector_field.png"):
    dx, dy = vector_field[..., 0], vector_field[..., 1]
    h, w = dx.shape
    x = np.arange(0, w, max(1, w // 30))
    y = np.arange(0, h, max(1, h // 30))
    xx, yy = np.meshgrid(x, y)

    if image is not None:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), alpha=0.5)

    # plt.axis('equal')
    plt.quiver(xx, yy, dx[y][:, x], dy[y][:, x])
    plt.xlim(-10, w + 10)
    plt.ylim(-10, h + 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(path)
    plt.clf()


def overlay_rgb_edges(rgb_image):
    h, w = rgb_image.shape[:2]
    dx_image = rgb_image[:, 1:w - 1] - rgb_image[:, 0:w - 2]
    dy_image = rgb_image[1:h - 1] - rgb_image[0:h - 2]
    drivative_map = np.maximum(dx_image[1:-1], dy_image[:, 1:-1]).max(-1)
    drivative_map = np.pad(drivative_map != 0, 1, mode='reflect').astype(np.uint8)

    # drivative_map = binary_dilation(drivative_map, iterations=1).astype(np.uint8)

    edges_map = np.ones_like(rgb_image) * 255
    edges_map[drivative_map != 0] = [0, 0, 0]

    return np.minimum(rgb_image, edges_map)


def plot_vornoi_cells(vornoi_map, points, oritentations, avoidance_map, path='vornoi_cells.png'):
    image = color.label2rgb(vornoi_map)
    image = overlay_rgb_edges(image)
    plt.imshow(image)
    plt.scatter(points[:, 1], points[:, 0], s=1, c='k')
    plt.imshow(avoidance_map * 255, alpha=0.5)

    plt.quiver(points[:, 1], points[:, 0], oritentations[:, 1], oritentations[:, 0])
    plt.savefig(path)
    plt.clf()


def get_rotation_matrix(theta):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    return np.array(((c, -s), (s, c)))


def aspect_ratio_resize(img, resize):
    return cv2.resize(img, (int(resize * img.shape[1] / img.shape[0]), resize))

