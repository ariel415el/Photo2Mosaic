import os

import cv2
import numpy as np
from tqdm import tqdm

from utils.tesselation.common import sample_centers, smooth_centers_by_gradients
from utils import plot_label_map, overlay_rgb_edges
from scipy.ndimage.measurements import label as scipy_label

def simplify_blobs(label_map, centers):
    """
    Convert the label of each blob to the label of its nearest center with onnected components
    """
    h, w = label_map.shape[:2]
    new_label_map = label_map.copy()
    yx_field = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=2)[..., ::-1]

    for i in range(len(centers)):
        blobs, ncomponents = scipy_label((label_map == i))
        blobs = [(blobs == i) for i in range(1, ncomponents + 1)]
        blobs = sorted(blobs, key=lambda x: x.sum())[::-1]
        for blob in blobs:
            blob_center = np.mean(yx_field[np.where(blob)], axis=0)
            dists = np.linalg.norm(centers - blob_center, axis=-1)
            blob_label = np.argmin(dists)
            new_label_map[np.where(blob)] = blob_label

    return new_label_map

def SLIC_superPixels(input_img, m, density_map, n_tiles, n_iters=10, search_area_factor=1, debug_dir=None):
    """
    Creates SLIC superpixels using spacial and color distances
    @param input_img: image to segment
    @param m: importance of space: smaller m means stricter color adherence
    @param density_map: unsigned integer map. The higher the number the denser will the cells in that area be
    @param density_map: unsigned integer map. The higher the number the denser will the cells in that area be
    @param n_tiles: approximately How many cells will there be
    @param n_iters: This is an iterative algorithm
    @param search_area_factor: To reduce memory and compute the search around each cell is restricted to small sorounding
    @return:
    """
    h, w = input_img.shape[:2]
    input_img = cv2.GaussianBlur(input_img, (5, 5), 5)

    centers = sample_centers((h, w), n_tiles, 'random', density_map)
    centers = smooth_centers_by_gradients(centers, cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY))

    lab_field = cv2.cvtColor(input_img, cv2.COLOR_BGR2Lab)
    yx_field = np.indices((h,w)).transpose(1,2,0)


    S = int(np.ceil(np.sqrt(h * w / n_tiles))) * search_area_factor
    for iter in tqdm(range(n_iters)):
        min_dist_map = np.inf * np.ones((len(centers), h, w))
        for c_idx in range(len(centers)):
            cy, cx = centers[c_idx]
            search_slice = tuple([slice(max(0, cy - S), cy + S), slice(max(0, cx - S), cx + S)])

            color_distances = ((lab_field[search_slice] - lab_field[cy, cx]) ** 2).sum(-1)
            spatial_distances = ((yx_field[search_slice] - yx_field[cy, cx]) ** 2).sum(-1)

            dist_map = color_distances / m + spatial_distances / S
            dist_map *= density_map[cy, cx]
            min_dist_map[c_idx][search_slice] = dist_map

        label_map = np.argmin(min_dist_map, axis=0)

        remove_indices = []
        for c_idx in range(len(centers)):
            if (label_map == c_idx).any():
                centers[c_idx] = np.mean(yx_field[label_map == c_idx], axis=0).astype(int)
            else:
                remove_indices.append(c_idx)

        centers = np.delete(centers, remove_indices, axis=0)

        if debug_dir:
            plot_label_map(label_map, centers, path=os.path.join(debug_dir, f'Clusters_{iter}.png'))

    label_map = simplify_blobs(label_map, centers)

    if debug_dir:
        plot_label_map(label_map, centers, centers, path=os.path.join(debug_dir, f'Connected_Clusters.png'))

    overlay_rgb_edges(input_img, label_map, path=os.path.join(debug_dir, f'Overlayed_clusters.png'))

    return centers, label_map

class SLIC:
    """SLIC method for superpixels as described in
    'SLIC Superpixels Compared to State-of-the-Art Superpixel Methods'
    Create kmeans using spatial and color features.
    """

    def __init__(self, n_tiles, m, init_mode):
        self.n_tiles = n_tiles
        self.init_mode = init_mode
        self.m = m



    def unfiy_blobs(self, label_map, h, w, S):
        """
        Makes sure all blobs are connected by changing the labels of isolated pixels.
        Possibly leads to more and different labels
        code from https://github.com/aleenawatson/SLIC_superpixels
        """
        new_label_map = -1 * np.ones((h, w)).astype(np.int64)
        elements = []
        label = 0
        adj_label = 0
        for i in range(w):
            for j in range(h):
                # Search for an un-assigned pixel
                if new_label_map[j, i] == -1:
                    elements = []
                    elements.append((j, i))

                    # If there is some assigned neighbor pixel use its label
                    for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                        x = elements[0][1] + dx
                        y = elements[0][0] + dy
                        if 0 <= x < w and 0 <= y < h and new_label_map[y, x] >= 0:
                            adj_label = new_label_map[y, x]

                # Flood same label recursively to all un-assigned pixels
                count = 1
                counter = 0
                while counter < count:
                    for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                        x = elements[counter][1] + dx
                        y = elements[counter][0] + dy

                        if 0 <= x < w and 0 <= y < h:
                            if new_label_map[y, x] == -1 and label_map[j, i] == label_map[y, x]:
                                elements.append((y, x))
                                new_label_map[y, x] = label
                                count += 1

                    counter += 1

                # If floding was local enough use neighbor label
                if count <= S / 2 and adj_label:
                    for counter in range(count):
                        new_label_map[elements[counter]] = adj_label

                    label -= 1

                label += 1

        # Repalce new blobs label with their existing nearest neighbors (spacialy only)
        # for c_idx in range(len(np.unique(label_map)), len(np.unique(new_label_map))):

        return new_label_map

    def tesselate(self, input_img, density_map=None, n_iters=10, debug_dir=None):
        h, w = input_img.shape[:2]
        input_img = cv2.GaussianBlur(input_img, (5, 5), 5)

        centers = sample_centers(input_img.shape[:2], self.n_tiles, self.init_mode, density_map=density_map)
        centers = smooth_centers(centers, cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY))

        lab_field = cv2.cvtColor(input_img, cv2.COLOR_BGR2Lab)
        xy_field = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=2)

        label_map = -1 * np.ones((h, w), dtype=int)
        min_dist_map = np.inf * np.ones((h, w))

        S = int(np.ceil(np.sqrt(h * w / self.n_tiles)))
        for iter in range(n_iters):
            print(iter)
            for c_idx in range(len(centers)):
                center = centers[c_idx]
                search_slice = tuple([slice(max(0, center[k] - S), center[k] + S) for k in range(2)])
                color_distances = ((lab_field[search_slice] - lab_field[center[0], center[1]]) ** 2).sum(-1)
                spatial_distances = ((xy_field[search_slice] - xy_field[center[0], center[1]]) ** 2).sum(-1)
                D = color_distances / self.m + spatial_distances / S
                condition = D < min_dist_map[search_slice]
                min_dist_map[search_slice][condition] = D[condition]
                label_map[search_slice][condition] = c_idx

            for c_idx in range(len(centers)):
                if (label_map == c_idx).any():
                    centers[c_idx] = np.mean(xy_field[label_map == c_idx], axis=0).astype(int)[::-1]
                else:
                    centers[c_idx] = np.random.random(2) * np.array((h, w)).astype(
                        np.uint64)  # TODO This is not so good

            if debug_dir:
                plot_label_map(label_map, centers, path=os.path.join(debug_dir, f'Clusters_{iter}.png'))
        if debug_dir:
            plot_label_map(self.simplify_blobs(label_map, centers), centers,
                           path=os.path.join(debug_dir, f'Connected_Clusters.png'))

        label_map = self.simplify_blobs(label_map, centers)

        utils.overlay_rgb_edges(input_img, label_map, path=os.path.join(debug_dir, f'Overlayed_clusters.png'))

        return centers, label_map