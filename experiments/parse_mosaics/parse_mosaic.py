import os

import cv2
import numpy as np

from utils.tesselation import SLIC
from skimage.color import label2rgb
from scipy import signal



if __name__ == '__main__':
    mosaic_reference_path = '/home/ariel/Downloads/mosaic7_crop.jpg'
    img = cv2.imread(mosaic_reference_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # detect_corners(img)
    show_gradient_map(img)

    # tesselate_mosaic(img, n_clusters=10)

    # flood_to_contour(img, (30, 30))
    # flood_to_contour(img, (35, 60))
