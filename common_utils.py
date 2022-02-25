import cv2
import numpy as np
from matplotlib import pyplot as plt

from skimage import color
from scipy.ndimage.morphology import binary_dilation


def plot_vector_field(vector_field, image=None, path="vector_field.png"):
    dx, dy = vector_field[..., 1], vector_field[..., 0].copy()
    h, w = dx.shape
    x = np.arange(0, w, max(1, w // 25))
    y = np.arange(0, h, max(1, h // 25))
    xx, yy = np.meshgrid(x, y)

    if image is not None:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), alpha=0.5)

    # plt.axis('equal')
    plt.quiver(xx, yy, dx[y][:, x], dy[y][:, x], angles='xy')
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

    plt.quiver(points[:, 1], points[:, 0], oritentations[:, 1], oritentations[:, 0], angles='xy')
    plt.savefig(path)
    plt.clf()


def get_rotation_matrix(theta):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    return np.array(((c, -s), (s, c)))


def aspect_ratio_resize(img, resize, mode=None):
    return cv2.resize(img, (int(resize * img.shape[1] / img.shape[0]), resize), interpolation=mode)


def image_histogram_equalization(image, number_bins=256):
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


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

def get_edges_map_canny(image, blur_size=5, sigma=2):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (blur_size, blur_size), sigma)

    # mask = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    mask = cv2.Canny(image=img_blur, threshold1=100, threshold2=150)

    mask[mask==255] = 1

    return mask


def normalize_vector_field(vector_field):
    """Divide an np aarray with last dimension of size 2 by its norm in this axis, replace zero divisions"""
    norms = np.linalg.norm(vector_field, axis=-1)
    vector_field /= norms[..., None]
    if 0 in norms:
        nans = np.where(norms == 0)
        random_direction = np.random.rand(len(nans[0]), 2)
        random_direction /= np.linalg.norm(random_direction, axis=1, keepdims=True)
        vector_field[nans] = random_direction

    return vector_field
