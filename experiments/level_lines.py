import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_closing, binary_opening, binary_erosion

if __name__ == '__main__':

    img = cv2.imread('/home/ariel/projects/Photo2Mosaic/Photo2Mosaic/images/images/Alexander.jpg', cv2.IMREAD_GRAYSCALE)
    k = 5
    img = cv2.GaussianBlur(img, (k, k), k)

    # b = 255 // 5
    # img = (img // b)*b
    # img += b // 2

    cv2.imshow("Asd", img)
    cv2.waitKey(0)
    level_map = np.zeros_like(img)

    for v in range(img.min() + 5, img.max(), 15):
        level_map[img == v] = 255

    level_map = binary_closing(level_map, iterations=1).astype(np.uint8)*255


    cv2.imshow("Asd", level_map)
    cv2.waitKey(0)

