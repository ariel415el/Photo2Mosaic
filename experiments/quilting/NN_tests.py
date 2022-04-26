import math

import cv2
import numpy as np
import torch

from GPNN_tiling.NN_modules import PytorchNNLowMemory
from GPNN_tiling.utils import cv2pt, extract_patches
import torch.nn.functional as F
import cv2

if __name__ == '__main__':
    content = cv2.imread('/home/ariel/Downloads/single_curve.jpg')
    texture = cv2.imread('/home/ariel/Downloads/mosaic7.jpg')

    center = [217, 135]


    content_patch = content