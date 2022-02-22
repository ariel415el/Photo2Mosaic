import cv2
import numpy as np
from matplotlib import pyplot as plt

from skimage import color
from scipy.ndimage.morphology import binary_dilation


def get_derivative_binary_mask(rgb_image):
    """returns a binary mask of edges in an RGB image"""
    h, w = rgb_image.shape[:2]
    dx_image = rgb_image[:, 1:w - 1] - rgb_image[:, 0:w - 2]
    dy_image = rgb_image[1:h - 1] - rgb_image[0:h - 2]
    drivative_map = np.maximum(dx_image[1:-1], dy_image[:, 1:-1]).max(-1)
    edges_map = np.pad(drivative_map != 0, 1, mode='reflect').astype(np.uint8)

    return edges_map


def plot_vector_field(reference_image, vector_field, path="vector_field.png"):
    dx, dy = vector_field[..., 0], vector_field[..., 1]
    h, w = dx.shape
    x = np.arange(0, w,  max(1, w // 30))
    y = np.arange(0, h, max(1, h // 30))
    xx, yy = np.meshgrid(x, y)

    plt.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB) , alpha=0.5)

    # plt.axis('equal')
    plt.quiver(xx, yy, dx[y][:, x], dy[y][:, x])
    plt.xlim(-10, w + 10)
    plt.ylim(-10, h + 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(path)
    plt.clf()


def overlay_edges(rgb_image):
    drivative_map = get_derivative_binary_mask(rgb_image)
    # drivative_map = binary_dilation(drivative_map, iterations=1)

    edges_map = np.ones_like(rgb_image)*255
    edges_map[drivative_map != 0] = [0,0,0]

    return np.minimum(rgb_image, edges_map)


def plot_vornoi_cells(vornoi_map, points, oritentations, avoidance_map, path='vornoi_cells.png'):
    image = color.label2rgb(vornoi_map)
    image = overlay_edges(image)
    plt.imshow(image)
    plt.scatter(points[:,1], points[:,0], s=1, c='k')
    plt.imshow(avoidance_map*255, alpha=0.5)

    plt.quiver(points[:,1], points[:,0], oritentations[:,1], oritentations[:,0])
    plt.savefig(path)
    plt.clf()


def render_tiles(centers, oritentations, image, tile_size, alpha_mask, path='mosaic.png'):
    mosaic = np.ones_like(image) * 127
    loss_image = np.zeros(image.shape[:2])

    corner_1_direction = (oritentations @ get_rotation_matrix(45))
    corner_2_direction = (oritentations @ get_rotation_matrix(135))
    corner_3_direction = (oritentations @ get_rotation_matrix(225))
    corner_4_direction = (oritentations @ get_rotation_matrix(315))

    tile_diameter = tile_size / np.sqrt(2)
    for i in range(centers.shape[0]):
        alpha = alpha_mask[centers[i][0], centers[i][1]]
        contours = [(centers[i] + corner_1_direction[i] * tile_diameter / alpha),
                    (centers[i] + corner_2_direction[i] * tile_diameter / alpha),
                    (centers[i] + corner_3_direction[i] * tile_diameter / alpha),
                    (centers[i] + corner_4_direction[i] * tile_diameter / alpha)]

        contours = np.array(contours)[:, ::-1]

        color = image[int(centers[i][0]), int(centers[i][1])]
        box = np.int0(cv2.boxPoints(cv2.minAreaRect(np.array(contours).astype(int))))
        cv2.drawContours(mosaic, [box], -1, color=color.tolist(), thickness=cv2.FILLED)
        # cv2.drawContours(mosaic, [box], -1, color=(0,0,0), thickness=2)

        tmp = np.zeros_like(loss_image)
        cv2.drawContours(tmp, [box], -1, color=1, thickness=cv2.FILLED)
        loss_image += tmp

    n_ovverided_pixels = loss_image[loss_image > 1].sum()
    loss = n_ovverided_pixels / loss_image.size
    plt.title(f"overided pixels: {n_ovverided_pixels} / {loss_image.size} (={loss * 100:.1f}%)")
    plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    return loss


def get_rotation_matrix(theta):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    return np.array(((c, -s), (s, c)))
