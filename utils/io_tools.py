import os

import cv2
import numpy as np

from utils.image import set_image_height_without_distortion


def load_images(img_path, size_map_path, resize):
    image = cv2.imread(img_path)
    if resize:
        image = set_image_height_without_distortion(image, resize)
    if size_map_path is not None:
        size_map = cv2.imread(size_map_path, cv2.IMREAD_GRAYSCALE)
        size_map[size_map == 255] = 2
        size_map[size_map == 0] = 1
        if resize:
            size_map = set_image_height_without_distortion(size_map, resize, mode=cv2.INTER_NEAREST)
    else:
        size_map = np.ones(image.shape[:2], dtype=int)
    assert image.shape[:2] == size_map.shape[:2]

    return image, size_map


def create_output_dirs(outputs_dir):
    debug_dir = os.path.join(outputs_dir, "debug")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)


