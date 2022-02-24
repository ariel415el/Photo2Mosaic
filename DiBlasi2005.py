import os.path
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening
from tqdm import tqdm
from matplotlib import pyplot as plt

from common_utils import plot_vornoi_cells, get_rotation_matrix, plot_vector_field, aspect_ratio_resize, \
    get_edges_map_canny

EDGE_LABEL = 1
FREE_SPACE_LABEL = 2


def create_gradient_and_level_matrix(edge_map, size_map, default_level_gap):
    """
    Compute distance transform from edges, return normalized gradient of distances
    Compute a level map from integer modolu of distances
    """
    dist_transform = ndimage.distance_transform_edt(edge_map == 0)

    direction_field = np.stack(np.gradient(dist_transform), axis=2)
    direction_field = cv2.GaussianBlur(direction_field, (5, 5), 0)

    norms = np.linalg.norm(direction_field, axis=2)
    direction_field /= norms[:, :, None]
    if 0 in norms:
        nans = np.where(norms == 0)
        random_direction = np.random.rand(len(nans[0]), 2)
        random_direction /= np.linalg.norm(random_direction, axis=1, keepdims=True)
        direction_field[nans[0], nans[1]] = random_direction

    dist_transform = dist_transform.astype(int)
    level_matrix = np.zeros_like(dist_transform).astype(np.uint8)
    for size in np.unique(size_map):
        gap_size = default_level_gap // size
        mask = np.remainder(dist_transform, gap_size) == gap_size // 2
        mask = np.logical_and(mask, size_map == size)
        level_matrix[mask] = FREE_SPACE_LABEL

    return direction_field, level_matrix


def oriented_distance(normal, handicap_factor, vector):
    """Compute l1 distance in oriented axis with handicap on the first normal"""
    normal2 = normal @ get_rotation_matrix(90)
    dist = handicap_factor * (vector @ normal)**2 + (vector @ normal2)**2
    return dist


def find_approximate_location(center, orientation, diameter, candidate_location_map, handicap_factor=2):
    """
    Look in the orientedd surrounding of 'center' for free space in 'candidate_location_map'
    """
    arange = np.arange(-diameter, diameter)
    rs, cs = np.meshgrid(arange, arange)
    surrounding_points = np.column_stack((rs.ravel(), cs.ravel()))
    # sort surrounding_points by an oriented distance disprefering points in the direction of the normal to the edge
    surrounding_points = sorted(list(surrounding_points), key=lambda x: oriented_distance(orientation, handicap_factor, x))
    for offset in surrounding_points:
        point = center + offset
        point = np.minimum(point, np.array(candidate_location_map.shape) - 1)
        point = np.maximum(point, np.array([0,0]))
        if candidate_location_map[point[0], point[1]] == FREE_SPACE_LABEL:
            return point
    return None


def get_tile_contour(center, corner_diameter, direction_field):
    orientation = direction_field[center[0], center[1]]
    corner_directions = orientation @ np.stack([get_rotation_matrix(45),
                                                get_rotation_matrix(135),
                                                get_rotation_matrix(225),
                                                get_rotation_matrix(315)])

    contours = center + corner_directions * corner_diameter
    contours = contours[:, ::-1]
    box = np.int0(cv2.boxPoints(cv2.minAreaRect(np.array(contours).astype(int))))

    return box


class Blasi05MosaicMaker:
    def __init__(self, configs):
        self.config = configs
        self.image = aspect_ratio_resize(cv2.imread(self.config.img_path), self.config.resize)
        self.size_map = np.ones(self.image.shape[:2])
        if self.config.size_map_path is not None:
            self.size_map = cv2.imread(self.config.size_map_path, cv2.IMREAD_GRAYSCALE)
            self.size_map[self.size_map == 0] = 1
            self.size_map[self.size_map == 255] = 2
            self.size_map = aspect_ratio_resize(self.size_map, self.config.resize, mode=cv2.INTER_NEAREST)
        assert self.image.shape[:2] == self.size_map.shape[:2]

    def _create_output_dirs(self, outputs_dir):
        self.debug_dir = os.path.join(outputs_dir, "debug")
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

    def make(self, outputs_dir):
        self._create_output_dirs(outputs_dir)
        
        # Create the matrices needed for buildin the mosaic
        edge_map = get_edges_map_canny(self.image)
        direction_field, level_matrix = create_gradient_and_level_matrix(edge_map, self.size_map, self.config.default_tile_size + self.config.extra_level_gap)

        # Debug code:
        plot_vector_field(direction_field, self.image, path=os.path.join(self.debug_dir, 'VectorField.png'))
        cv2.imwrite(os.path.join(self.debug_dir, 'EdgeMap.png'), edge_map * 255)
        cv2.imwrite(os.path.join(self.debug_dir, 'Level_matrix.png'), level_matrix * 127)

        self._render_tiles(direction_field, level_matrix, path=os.path.join(outputs_dir, f'FinalMosaic.png'))

    def _render_tiles(self, direction_field, level_matrix, path='FinalMosaic.png'):
        """
        Render the mosaic: place oritented squares on a canvas with size defined by the alpha mask
        """
        mosaic = np.ones_like(self.image) * 127
        candidate_location_map = level_matrix.copy()
        coverage_map = np.zeros(self.image.shape[:2])


        pbar = tqdm()
        n_placed_tiles = 0
        center = None
        while (candidate_location_map == 2).any():
            if center is None:  # find a random free-space
                candidate_points = np.stack(np.where(candidate_location_map == 2), axis=1)
                center = candidate_points[np.random.randint(0, len(candidate_points))]
            else:  # Find an adjecnt place on the chain
                orientation = direction_field[center[0], center[1]]
                search_area_size = self.config.default_tile_size * 2 * self.size_map[center[0], center[1]]
                center = find_approximate_location(center, orientation, search_area_size, candidate_location_map, handicap_factor=2)
                if center is None:
                    continue

            corner_diameter = (self.config.default_tile_size / np.sqrt(2)) / self.size_map[center[0], center[1]]

            tile_color = self.image[int(center[0]), int(center[1])].tolist()
            contour = get_tile_contour(center, corner_diameter, direction_field)
            cv2.drawContours(mosaic, [contour], -1, color=tile_color, thickness=cv2.FILLED)
            # cv2.drawContours(mosaic, [contour], -1, color=(0, 0, 0), thickness=1)

            # update coverage_map
            tmp = np.zeros_like(coverage_map)
            cv2.drawContours(tmp, [contour], -1, color=1, thickness=cv2.FILLED)
            coverage_map += tmp

            # Remove surrounding from candidate points
            delete_diameter = corner_diameter * self.config.gap_factor
            contour = get_tile_contour(center, delete_diameter, direction_field)
            cv2.drawContours(candidate_location_map, [contour], -1, color=0, thickness=cv2.FILLED)

            n_placed_tiles += 1
            pbar.set_description(f"Placed tiles: {n_placed_tiles}")

            if n_placed_tiles % 100 == 0:
                tmp = mosaic + ((candidate_location_map == 2).astype(np.uint8)*255*0.4)[:,:,None]
                cv2.imwrite(os.path.join(self.debug_dir, f"Moisaic-step-{n_placed_tiles}.png"), tmp)

        coverage_percentage = (coverage_map > 0).sum() / coverage_map.size * 100
        overlap_percentage = (coverage_map > 1).sum() / coverage_map.size * 100
        print(f"Coverage: {coverage_percentage:.1f}% Overlap:{overlap_percentage:.1f}%)")
        plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.savefig(path)
        plt.clf()


@dataclass
class MosaicConfig:
    img_path: str = 'images/YingYang.png'
    size_map_path: str = 'images/YingYang_mask.png'
    resize: int = 512
    default_tile_size = 15
    gap_factor = 2  # determines the size of the minimal gap between placed tiles (multiplies the tile diameter)
    extra_level_gap = 2  # the gap between level lines will be tile_size + levels_gap
    initial_location: str = 'uniform' # random / uniform

    def get_str(self):
        im_name = os.path.basename(os.path.splitext(self.img_path)[0])
        return f"{im_name}_R-{self.resize}_T-{self.default_tile_size}"


if __name__ == '__main__':
    device: torch.device = torch.device("cpu")
    configs = MosaicConfig()
    mosaic_maker = Blasi05MosaicMaker(configs)
    mosaic_maker.make(os.path.join("DiBlasi2005_outputs", configs.get_str()))

