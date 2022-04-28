import lic

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import color


def plot_vector_field(vector_field, image=None, path="vector_field.png"):
    dx, dy = vector_field[..., 1], vector_field[..., 0].copy()
    h, w = dx.shape
    x = np.arange(0, w, max(1, w // 30))
    y = np.arange(0, h, max(1, h // 30))
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


def overlay_rgb_edges(image, label_map, path):
    h, w = image.shape[:2]
    dx_image = label_map[:, 1:w - 1] - label_map[:, 0:w - 2]
    dy_image = label_map[1:h - 1] - label_map[0:h - 2]
    drivative_map = np.maximum(dx_image[1:-1], dy_image[:, 1:-1])
    drivative_map = np.pad(drivative_map != 0, 1, mode='reflect').astype(np.uint8)

    # drivative_map = binary_dilation(drivative_map, iterations=1).astype(np.uint8)

    new_image = image.copy()
    new_image[drivative_map != 0] = [0, 0, 0]

    cv2.imwrite(path, new_image)

    return new_image

COLORS = None
def plot_label_map(label_map, points=None, oritentations=None, avoidance_map=None, path='vornoi_cells.png'):
    global COLORS
    if points is not None:
        if COLORS is None:
            n = len(points)
            COLORS = [tuple(np.random.choice(np.linspace(0,1,255), size=3)) for _ in range(n)]

    image = color.label2rgb(label_map, colors=COLORS)

    # for i in range(len(points)):
    #     image = cv2.putText(image, str(i),points[i][::-1],cv2.FONT_HERSHEY_SIMPLEX,0.25,(0,0,255),1,2)

    w, h = image.shape[:2]
    plt.figure(figsize=(h / 50, w / 50))
    plt.imshow(image)

    if points is not None:
        plt.scatter(points[:, 1], points[:, 0], s=4, c='r')

    if avoidance_map is not None:
        plt.imshow(avoidance_map * 255, alpha=0.5)

    if oritentations is not None:
        plt.quiver(points[:, 1], points[:, 0], oritentations[:, 1], oritentations[:, 0], angles='xy')

    plt.tight_layout()
    plt.savefig(path)
    plt.clf()


def line_interval_convolution(direction_field, path=None):
    """visualizes a vector field"""
    lic_result = lic.lic(direction_field[...,0], direction_field[...,1], length=32)
    plt.imshow(lic_result, origin='lower', cmap='gray')
    plt.gca().invert_yaxis()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()
