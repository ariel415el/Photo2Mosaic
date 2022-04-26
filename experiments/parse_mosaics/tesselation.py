import os

from skimage.color import label2rgb
from tensorflow_datasets.object_detection.open_images_challenge2019_beam import cv2

from utils.tesselation import SLIC


def tesselate_mosaic(img, n_clusters):
    tesselator = SLIC(n_clusters, m=1, init_mode='random')
    os.makedirs("./tesselation", exist_ok=True)
    centers, label_map = tesselator.tesselate(img, debug_dir="./tesselation")

    cv2.imshow("cells", label2rgb(label_map))
    cv2.waitKey(0)
