import os.path

import cv2
import numpy as np

from GPNN_tiling.utils import aspect_ratio_resize
from experiments.parse_mosaics.detect_lines import n_channels_gradient
from experiments.quantize_image import show_images

def canny(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(image=img, threshold1=100, threshold2=200)
    return edges

def np_gradient(img):
    magnitudes, angles = n_channels_gradient(img)
    edges = magnitudes.astype(np.uint8)
    return edges


def DOG(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(img, (21, 21), 0)
    g2 = cv2.GaussianBlur(img, (23, 23), 0)

    return g1 - g2

if __name__ == '__main__':
    path = '/home/ariel/university/GPDM/images/mosaics/mosaic7.jpg'
    # path = '/home/ariel/projects/Photo2Mosaic/Photo2Mosaic/images/images/YingYang.png'
    # path = '/home/ariel/Downloads/mosaic7_crop.jpg'
    img = cv2.imread(path)
    # img = cv2.imread('/home/ariel/Downloads/File-4_crop.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    edge_maps = []



    edges = canny(img)
    edge_maps.append((edges, 'Canny'))

    edges = np_gradient(img)
    edge_maps.append((edges, 'NP'))

    edges = DOG(img)
    edge_maps.append((edges, 'DOG'))

    show_images(img, edge_maps)
