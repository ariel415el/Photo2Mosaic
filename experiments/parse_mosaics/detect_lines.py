import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_opening

from skimage.color import label2rgb

from experiments.parse_mosaics.utils import show_images


def cv_LSD(img):
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)

    # lines = np.concatenate([lsd.detect(img[..., i])[0] for i in range(3)])

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = lsd.detect(gray_img)[0]

    for i in range(len(lines)):
        start_x, start_y, end_x, end_y = lines[i, 0]
        if np.linalg.norm(np.array([start_x, start_y]) - np.array([end_x, end_y])) > 5:
            cv2.line(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0,255,255),thickness=1, lineType=cv2.LINE_AA)

    h, w = img.shape[:2]

    cv2.imshow("LSD", img)
    cv2.waitKey(0)

def n_channels_gradient(img):
    img = img.astype(np.float32)
    dx, dy = img.copy(), img.copy()
    for i in range(img.shape[-1]):
        dx[..., i], dy[..., i] = np.gradient(img[..., i])

    magnitudes = np.sqrt(dx**2 + dy**2).mean(-1)
    angles = 0.5 * np.arctan2(dy, dx).mean(-1)

    return magnitudes, angles



def grow_angle_regions(angle_map, mask=None, angle_percision_deg=np.deg2rad(22.5), region_min_size=20):
    if mask is None:
        mask = np.ones_like(angle_map)
    h, w = angle_map.shape
    region_labels = np.zeros((h, w))
    next_label = 1
    region = []
    last_coords_batch = []

    while np.any(mask):
        if not last_coords_batch:
            # mark region
            if region:
                np_region = np.array(region)
                if len(np_region) > region_min_size:
                    region_labels[np_region[:, 0], np_region[:, 1]] = next_label
                    next_label += 1

            # start new region
            unvisited_coordinates = np.stack(np.where(mask), -1)
            random_coord = unvisited_coordinates[np.random.choice(np.arange(len(unvisited_coordinates)))]
            region = last_coords_batch = [random_coord]

        np_region = np.array(region)
        cur_region_angle = angle_map[np_region[:, 0], np_region[:, 1]].mean()

        new_coordinate_batch = []
        for coord in last_coords_batch:
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    if (0 <= coord[1] + dx < w) and (0 <= coord[0] + dy < h):
                        candidate_coord = [coord[0] + dy, coord[1] + dx]
                        candidate_angle = angle_map[candidate_coord[0], candidate_coord[1]]
                        is_unvisited = mask[candidate_coord[0], candidate_coord[1]] == 1
                        is_similar = np.abs(candidate_angle - cur_region_angle) < angle_percision_deg
                        if is_unvisited and is_similar:
                            mask[candidate_coord[0], candidate_coord[1]] = 0
                            new_coordinate_batch.append(candidate_coord)
        region += last_coords_batch
        last_coords_batch = new_coordinate_batch
    return region_labels

def gradient_region_growing(img, gradient_t=15, angle_t=15, region_size_t=2):

    img = cv2.GaussianBlur(img, (5,5), 0.5)

    magnitudes, angles = n_channels_gradient(img)

    # angles = cv2.GaussianBlur(angles, (7,7), 1)
    # color_angles = (plt.get_cmap('copper')(angles)[...,:-1] * 255).astype(np.uint8)
    # plt.imshow(np.rad2deg(angles))
    # plt.show()

    mask = magnitudes.copy()
    mask[mask < gradient_t] = 0
    mask[mask >= gradient_t] = 1

    # mask = cv2.Canny(image=cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY), threshold1=100, threshold2=200)
    # mask[mask == 255] = 1

    region_labels = grow_angle_regions(angles, mask.copy(), np.deg2rad(angle_t))

    regions = label2rgb(region_labels)

    regions[mask == 0] = 0
    regions[region_labels == 0] = 0

    show_images([
        (img, "img"),
        (mask, f"Edge mask (t{gradient_t})"),
        (regions, f"regions (angle_t={angle_t}, region_t={region_size_t})"),
    ])


if __name__ == '__main__':
    img = cv2.imread('/home/ariel/Downloads/mosaic7_crop.jpg')
    img = cv2.imread('/home/ariel/projects/Photo2Mosaic/Photo2Mosaic/images/images/Alexander.jpg')
    h,w = img.shape[:2]
    img = cv2.resize(img, (int(w*0.5), int(h*0.5)))
    # img = cv2.imread('/home/ariel/Downloads/File-4_crop.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gradient_region_growing(img,gradient_t=20,angle_t=15, region_size_t=5)
    # cv_LSD(img)



