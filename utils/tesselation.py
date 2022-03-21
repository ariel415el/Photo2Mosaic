import os

import cv2
import numpy as np
import torch
from scipy.ndimage.measurements import label as scipy_label

import utils
from utils import get_rotation_matrix, plot_label_map, set_image_height_without_distortion

ROTATION_MATRICES={x: get_rotation_matrix(x) for x in [45, 90, -45]}





class VornoiTessealtion:
    """
    iteratively map each point
    """
    def __init__(self, n_tiles, density_map, init_mode, torch_device=torch.device("cpu")):
        self.density_map = density_map
        self.centers = sample_centers(density_map.shape[:2], n_tiles, init_mode, density_map)
        self.oritentations = sample_random_normals(n_tiles)
        self.device = torch_device

    def update_point_locations(self, direction_field, avoidance_map):
        """
        Iterative approach for computing centridal Vornoi cells using brute force equivalet of the Z-buffer algorithm
        For point in "points" compute a distance map now argmin over all maps to get a single map with index of
        closest point to each coordinate.
        The metric is used is the L1 distance with axis rotated by the points' orientation
        """

        h, w = direction_field.shape[:2]
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        coords = np.stack([yy, xx], axis=-1)  # coords[a ,b] = [b,a]

        diffs = coords - self.centers[:, None, None]

        basis1 = (self.oritentations @ ROTATION_MATRICES[-45])[:, None, None]
        basis2 = (self.oritentations @ ROTATION_MATRICES[45])[:, None, None]

        # Heavy computation on GPU
        with torch.no_grad():
            gaps = torch.from_numpy(diffs).to(self.device)
            basis1 = torch.from_numpy(basis1).to(self.device)  # u
            basis2 = torch.from_numpy(basis2).to(self.device)  # v

            # L1_u-v((x,y),(x0,y0)) = |<(x,y),u> - <(x0,y0),u>| + |<(x,y),v> - <(x0,y0),v>| = |<(x,y)-(x0,y0),u>| + |<(x,y)-(x0,y0),u>|
            distance_maps = torch.abs((gaps * basis1).sum(-1)) + torch.abs((gaps * basis2).sum(-1))
            distance_maps = distance_maps.cpu().numpy()

        for v in np.unique(self.density_map):
            distance_maps[self.density_map[self.centers[:, 0], self.centers[:, 1]] == v] *= v

        vornoi_map = np.argmin(distance_maps, axis=0)

        # Mark edges as ex-teritory (push cells away from edges)
        vornoi_map[avoidance_map == 1] = vornoi_map.max() + 2

        # ensure non-empty cells (points may not be closest to itself since argmax between same values is arbitraray)
        vornoi_map[self.centers[:, 0], self.centers[:, 1]] = np.arange(len(self.centers))

        # Move points to be the centers of their according vornoi cells
        self.centers = np.array([coords[vornoi_map == i].mean(0) for i in range(len(self.centers))]).astype(np.uint64)

        # update orientations using vector field
        self.oritentations = direction_field[self.centers[:, 0].astype(int), self.centers[:, 1].astype(int)]

        return self.centers, self.oritentations, vornoi_map

    def tesselate(self, direction_field, avoidance_map, n_iters, debug_dir=None):
        centers = oritentations = vornoi_map = None
        for i in range(n_iters):
            centers, oritentations, vornoi_map = self.update_point_locations(direction_field, avoidance_map)
            if debug_dir:
                plot_label_map(vornoi_map, centers, oritentations, avoidance_map, path=os.path.join(debug_dir, f'Vornoi_diagram_{i}.png'))
        return centers, oritentations, vornoi_map


class SLIC:
    """SLIC method for superpixels as described in
    'SLIC Superpixels Compared to State-of-the-Art Superpixel Methods'
    Create kmeans using spatial and color features.
    """
    def __init__(self, n_tiles, m, init_mode):
        self.n_tiles = n_tiles
        self.init_mode = init_mode
        self.m = m

    def simplify_blobs(self, label_map, centers):
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

    def unfiy_blobs(self, label_map,  h, w, S):
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
                search_slice = tuple([slice(max(0, center[k]-S), center[k]+S) for k in range(2)])
                color_distances = ((lab_field[search_slice] - lab_field[center[0], center[1]])**2).sum(-1)
                spatial_distances = ((xy_field[search_slice] - xy_field[center[0], center[1]])**2).sum(-1)
                D = color_distances / self.m + spatial_distances / S
                condition = D < min_dist_map[search_slice]
                min_dist_map[search_slice][condition] = D[condition]
                label_map[search_slice][condition] = c_idx

            for c_idx in range(len(centers)):
                if (label_map == c_idx).any():
                    centers[c_idx] = np.mean(xy_field[label_map == c_idx], axis=0).astype(int)[::-1]
                else:
                    centers[c_idx] = np.random.random(2) * np.array((h,w)).astype(np.uint64) # TODO This is not so good

            if debug_dir:
                plot_label_map(label_map, centers, path=os.path.join(debug_dir, f'Clusters_{iter}.png'))
        if debug_dir:
            plot_label_map(self.simplify_blobs(label_map, centers), centers, path=os.path.join(debug_dir, f'Connected_Clusters.png'))

        label_map = self.simplify_blobs(label_map, centers)

        utils.overlay_rgb_edges(input_img, label_map, path=os.path.join(debug_dir, f'Overlayed_clusters.png'))

        return centers, label_map


def sample_centers(image_shape, N, init_mode, density_map=None):
    """Sample initial tile centers and orientations  cell centers and directions"""
    h, w = image_shape
    if init_mode == 'random' and density_map is not None:  # sample randomly in each are in proportion to tile sizes in it
        dense_area_proportion = (density_map == 2).sum() / density_map.size

        # if X is the num_dense_area_points then the sparse area has X/4 and solving leads to
        num_dense_area_points = int(4 * dense_area_proportion * N / (1 + 4 * dense_area_proportion))
        num_default_area_points = N - num_dense_area_points

        posible_points = np.stack(np.where(density_map == 2), axis=1)
        dense_positions = posible_points[np.random.randint(len(posible_points), size=num_dense_area_points)]
        posible_points = np.stack(np.where(density_map == 1), axis=1)
        default_positions = posible_points[np.random.randint(len(posible_points), size=num_default_area_points)]
        centers = np.concatenate([default_positions, dense_positions], axis=0).astype(int)

    else:  # uniform
        # s = int(np.ceil(np.sqrt(N)))
        # y, x = np.meshgrid(np.linspace(1, h - 1, s), np.linspace(1, w - 1, s))
        # centers = np.stack([y.flatten(), x.flatten()], axis=1).astype(np.uint64)[:N]

        S = int(np.ceil(np.sqrt(h * w / N)))
        y, x = np.meshgrid(np.arange(S // 2 - 1, h - S // 2 + 1, S), np.arange(S // 2 - 1, w - S // 2 + 1, S))
        centers = np.stack([y.flatten(), x.flatten()], axis=1)

    return centers

def sample_random_normals(N):
    oritentations = np.random.rand(N, 2)
    oritentations /= np.linalg.norm(oritentations, axis=1, keepdims=True)
    return oritentations


def smooth_centers(centers, gray_image):
    """
    Move cluster centers to the lowest gradient position in a 3x3 neighborhood.
    """
    gradient_magnitudes = np.linalg.norm(np.stack(np.gradient(gray_image), axis=-1), axis=-1)
    gradient_magnitudes = np.pad(gradient_magnitudes, 1, 'constant', constant_values=np.inf)
    for i in range(len(centers)):
        r, c = centers[i]
        neighberhood = gradient_magnitudes[r :r + 3, c:c + 3] # r -> r+1 because of padding
        argmin = np.where(neighberhood == np.amin(neighberhood))
        centers[i] = (r + argmin[0][0] - 1, c + argmin[1][0] -1)  # -1 for offset from center
    return centers