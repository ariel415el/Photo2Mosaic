import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize

from utils import set_image_height_without_distortion, simplify_contour
from scipy.ndimage.measurements import label as scipy_label


def show_contours(img, contours, color=0):
    new = np.ones_like(img)*255
    new = cv2.drawContours(new, contours, -1, color=color, thickness=1)

    cv2.imshow("asd", new)
    cv2.waitKey(0)

def debug_contour_points(contours):
    for cnt in contours:
        sc = plt.scatter(cnt[:, 0, 1], cnt[:, 0, 0], c=np.linspace(0, 255, len(cnt)))
        plt.colorbar(sc)
        plt.show()

def simplify_all_contours(contours, cutoff=0.001):
    new_contours = []
    for cnt in contours:
        new_contours.append(simplify_contour(cnt, cutoff))

    return new_contours


def sort_points_clockwize(points):

    mean = np.mean(points, axis=0)
    zero_mean_points = points - mean

    r = np.linalg.norm(zero_mean_points, axis=1)

    angles = np.where(zero_mean_points[:,0] > 0, np.arccos(zero_mean_points[:,1]/r), 2*np.pi-np.arccos(zero_mean_points[:,1]/r))

    mask = np.argsort(angles)

    points = points[mask]

    return points

def counter_clockwise_sort(points):
    return np.array(sorted(points, key=lambda point: point[1] * (-1 if point[0] >= 0 else 1)))

def get_contours(img, use_cv):
    if use_cv:
        contours, _ = cv2.findContours((img == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

    structure = np.ones((3, 3), dtype=np.int)
    blobs, ncomponents = scipy_label((img == 255), structure)

    contours = []
    for i in range(1, ncomponents):
        mask = blobs == i
        points = np.stack(np.where(mask), axis=-1)
        # points = sort_points_clockwize(points)
        cnt = points[...,None, ::-1]
        contours.append(cnt)

    return tuple(contours)

if __name__ == '__main__':
    img = cv2.imread('../../images/edge_maps/turk_edges.png', cv2.IMREAD_GRAYSCALE)

    # img = set_image_height_without_distortion(img, 512)

    # img[img < 50] = 0
    # img[img >= 50] = 255
    #
    # cv2.imshow("sketch", img)
    # cv2.waitKey(0)
    #
    # img = skeletonize(img // 255).astype(np.uint8)*255
    #
    # cv2.imshow("skeleton", img)
    # cv2.waitKey(0)

    # contours = get_contours(img, use_cv=False)
    contours = get_contours(img, use_cv=False)

    debug_contour_points(contours)

    show_contours(np.ones_like(img)*255, contours, color=0)