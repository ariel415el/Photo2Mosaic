import cv2
import numpy as np
from matplotlib import pyplot as plt

def nms(points, scores, proximity_threshold):
    indices = np.arange(len(scores))
    sorted_indices = np.argsort(scores)[::-1]
    for i in sorted_indices:
        dists = np.linalg.norm(points[indices[indices != i]] - points[i], axis=1)
        if dists.min() < proximity_threshold:
            indices = indices[indices != i]

    return points[indices]

def detect_corners(img):
    h,w = img.shape[:2]


    # R_values = np.mean([cv2.cornerHarris(img[...,i].astype(np.float32), 12, 5, 0.04) for i in range(3)], axis=0)
    R_values = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), 3, 3, 0.04)

    # img[R_values > 0.05 * R_values.max()] = [0, 0, 255]

    nwhere = np.where(R_values > 0.01 * R_values.max())
    R_values = R_values[nwhere]
    points = np.stack(nwhere, axis=1)

    # img = cv2.resize(img, (w*5, h*5))

    # points = nms(points, R_values, proximity_threshold=5)

    colors = (R_values - R_values.min()) / R_values.max() * 255


    for i in range(len(points)):
        img = cv2.circle(img, points[i][::-1], radius=2, thickness=cv2.FILLED, color=[0, int(colors[i]), 0])


    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('/home/ariel/Downloads/mosaic7_crop.jpg')
    img = cv2.imread('/home/ariel/Downloads/File-4_crop.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detect_corners(img)