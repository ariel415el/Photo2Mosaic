import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_edges(img):
    blur_size = 3
    sigma = 1
    t1, t2 = 500, 1000

    img = cv2.GaussianBlur(img, (blur_size, blur_size), sigma)

    img_color_space = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    edge_maps = [cv2.Canny(image=img_color_space[..., i], threshold1=t1, threshold2=t2, apertureSize=5) for i in range(3)]
    edge_maps = np.stack(edge_maps, axis=-1).max(-1)

    return edge_maps


def show_images(images):
    n = len(images)
    plt.figure(figsize=(n * 4, 4))
    for i, (img, name) in enumerate(images):
        plt.subplot(1, n, 1 + i)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    plt.clf()