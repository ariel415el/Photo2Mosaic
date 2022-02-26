import os.path
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm
from matplotlib import pyplot as plt

from common_utils import get_rotation_matrix, plot_vector_field, aspect_ratio_resize, \
    get_edges_map_canny, normalize_vector_field

EDGE_LABEL = 1
FREE_SPACE_LABEL = 2

ROTATION_MATRICES={x: get_rotation_matrix(x) for x in [45,90]}

def oriented_distance(normal, handicap_factor, vector):
    """Compute l1 distance in oriented axis with handicap on the first normal"""
    normal2 = normal @ ROTATION_MATRICES[90]
    dist = handicap_factor * np.abs(vector @ normal) + np.abs(vector @ normal2)
    return dist


def find_approximate_location(center, normal, diameter, candidate_location_map, search_area_normal_factor):
    """
    Look in the orientedd surrounding of 'center' for free space in 'candidate_location_map'
    """
    # arange = np.arange(diameter // 3, diameter)
    # arange = np.concatenate([-arange, arange])
    arange = np.arange(-diameter, diameter)
    rs, cs = np.meshgrid(arange, arange)
    all_offsets = np.column_stack((rs.ravel(), cs.ravel()))

    # sort surrounding_points by an oriented distance disprefering points in the direction of the normal to the edge
    all_offsets = sorted(list(all_offsets), key=lambda x: oriented_distance(normal, search_area_normal_factor, x))

    for offset in all_offsets:
        point = center + offset
        point = np.minimum(point, np.array(candidate_location_map.shape) - 1)
        point = np.maximum(point, np.array([0,0]))
        if candidate_location_map[point[0], point[1]] == FREE_SPACE_LABEL:
            return point
    return None


def get_tile_rectangle_contour(center, tile_size, normal, normal_factors=(1, 1)):
    """Get a cv2 drawable contour of a rectangle centered in 'center' with 2 faces in normal1, normal2"""
    vec1 = normal * normal_factors[0]
    vec2 = normal @ ROTATION_MATRICES[90] * normal_factors[1]
    corner_directions = np.array([vec1 + vec2,
                                  vec1 - vec2,
                                  -vec1 - vec2,
                                  -vec1 + vec2])

    contours = center + corner_directions * (tile_size / 2)
    contours = contours[:, ::-1]
    box = np.int0(cv2.boxPoints(cv2.minAreaRect(np.array(contours).astype(int))))

    return box


class Blasi05MosaicMaker:
    def __init__(self, configs):
        self.config = configs
        self.image = aspect_ratio_resize(cv2.imread(self.config.img_path), self.config.resize)
        if self.config.size_map_path is not None:
            self.size_map = cv2.imread(self.config.size_map_path, cv2.IMREAD_GRAYSCALE)
            self.size_map[self.size_map == 0] = 1
            self.size_map[self.size_map == 255] = 2
            self.size_map = aspect_ratio_resize(self.size_map, self.config.resize, mode=cv2.INTER_NEAREST)
        else:
            self.size_map = np.ones(self.image.shape[:2], dtype=int)
        assert self.image.shape[:2] == self.size_map.shape[:2]

    def _create_output_dirs(self, outputs_dir):
        self.debug_dir = os.path.join(outputs_dir, "debug")
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

    def make(self, outputs_dir):
        self._create_output_dirs(outputs_dir)
        
        # Create the matrices needed for buildin the mosaic
        if self.config.edges_reference == 'mask' and self.config.size_map_path is not None:
            edge_map = get_edges_map_canny(cv2.cvtColor(self.size_map*127, cv2.COLOR_GRAY2BGR))
        else:
            edge_map = get_edges_map_canny(self.image)

        direction_field, level_matrix = self._create_gradient_and_level_matrix(edge_map, self.config.direction_reference)

        # Debug code:
        plot_vector_field(direction_field, edge_map*255, path=os.path.join(self.debug_dir, 'VectorField.png'))
        cv2.imwrite(os.path.join(self.debug_dir, 'EdgeMap.png'), edge_map * 255)
        cv2.imwrite(os.path.join(self.debug_dir, 'Level_matrix.png'), level_matrix * 127)

        self._render_tiles(direction_field, level_matrix, path=os.path.join(outputs_dir, f'FinalMosaic.png'))

    def _get_level_matrix_from_dist_map(self, dist_map, offset=0.5):
        dist_map = dist_map.astype(int)
        level_matrix = np.zeros_like(dist_map).astype(np.uint8)
        default_levels_gap = self.config.default_tile_size + self.config.extra_level_gap
        for factor in np.unique(self.size_map):
            gap_size = default_levels_gap // factor
            mask = np.remainder(dist_map, gap_size) == int(offset * gap_size)
            mask = np.logical_and(mask, self.size_map == factor)
            level_matrix[mask] = FREE_SPACE_LABEL
        return level_matrix

    def _create_gradient_and_level_matrix(self, edge_map, direction_reference):
        """
        Compute distance transform from edges, return normalized gradient of distances
        Compute a level map from integer modolu of distances
        direction_reference: 'level_map'/'edges'. what binary mask will be used to compute the direction field
        """
        dist_transform = ndimage.distance_transform_edt(edge_map == 0)

        level_matrix = self._get_level_matrix_from_dist_map(dist_transform, offset=0.5)
        dist_transform = ndimage.distance_transform_edt(level_matrix == 0)
        level_matrix = self._get_level_matrix_from_dist_map(dist_transform, offset=0)

        if direction_reference == 'edges':
            dist_transform = ndimage.distance_transform_edt(edge_map == 0)
        else:
            dist_transform = ndimage.distance_transform_edt(level_matrix == 0)

        dist_transform = cv2.GaussianBlur(dist_transform, (5, 5), 2)
        direction_field = np.stack(np.gradient(dist_transform), axis=2)
        direction_field = normalize_vector_field(direction_field)

        return direction_field, level_matrix

    def _get_next_tile_position(self, position, direction_field, candidate_location_map):
        if position is not None:
            normal = direction_field[position[0], position[1]]
            search_area_size = (self.config.default_tile_size // self.size_map[position[0], position[1]]) + 1
            position = find_approximate_location(position, normal, search_area_size, candidate_location_map, self.config.search_area_normal_factor)

        if position is None:  # find a random free-space
            candidate_points = np.stack(np.where(candidate_location_map == 2), axis=1)
            position = candidate_points[np.random.randint(0, len(candidate_points))]

        return position

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
            center = self._get_next_tile_position(center, direction_field, candidate_location_map)

            local_tile_size = self.config.default_tile_size / self.size_map[center[0], center[1]]
            normal = direction_field[center[0], center[1]]

            contour = get_tile_rectangle_contour(center, local_tile_size, normal, (1, self.config.aspect_ratio))

            # draw oriented tile
            tile_color = self.image[int(center[0]), int(center[1])].tolist()
            cv2.drawContours(mosaic, [contour], -1, color=tile_color, thickness=cv2.FILLED)
            cv2.drawContours(mosaic, [contour], -1, color=(0, 0, 0), thickness=1)

            # update coverage_map
            tmp = np.zeros_like(coverage_map)
            cv2.drawContours(tmp, [contour], -1, color=1, thickness=cv2.FILLED)
            coverage_map += tmp

            # Remove surrounding from candidate points
            delete_area = local_tile_size * self.config.gap_factor
            contour = get_tile_rectangle_contour(center, delete_area, normal, (self.config.delete_area_normal_factor, self.config.aspect_ratio))
            cv2.drawContours(candidate_location_map, [contour], -1, color=0, thickness=cv2.FILLED)

            n_placed_tiles += 1
            pbar.set_description(f"Placed tiles: {n_placed_tiles}")

            if n_placed_tiles % 1 == 0:
                tmp = mosaic + ((candidate_location_map == 2).astype(np.uint8) * 255 * 0.4)[:,:,None]
                # Debug Draw tile normal
                center_xy = center[::-1].astype(int)
                direction_xy = (normal[::-1] * self.config.default_tile_size / 2).astype(int)
                cv2.arrowedLine(tmp, tuple(center_xy), tuple(center_xy + direction_xy), (0, 0, 255), 1)
                cv2.imwrite(os.path.join(self.debug_dir, f"Moisaic-step-{n_placed_tiles}.png"), tmp)

        coverage_percentage = (coverage_map > 0).sum() / coverage_map.size * 100
        overlap_percentage = (coverage_map > 1).sum() / coverage_map.size * 100
        print(f"Coverage: {coverage_percentage:.1f}% Overlap:{overlap_percentage:.1f}%)")
        cv2.imwrite(path, mosaic)


@dataclass
class MosaicConfig:
    img_path: str = 'images/Elon.jpg'
    size_map_path: str = 'images/Elon_mask.png'
    resize: int = 512
    default_tile_size = 16
    gap_factor = 2  # determines the size of the minimal gap between placed tiles (multiplies the tile diameter)
    extra_level_gap = 2  # the gap between level lines will be tile_size + levels_gap
    direction_reference = 'edges'  # edges/level_map: use edges or levelmap for directions field
    edges_reference = 'mask'  # image/mask compute edges from image itself or from the mask
    aspect_ratio = 1
    search_area_normal_factor: int = 1000      # how much to decrease the local search area along the first normal
    delete_area_normal_factor: float = 0.9  # how much to decrease the area of deleted open slots in the direction of the first nomal

    def get_str(self):
        im_name = os.path.basename(os.path.splitext(self.img_path)[0])
        return f"{im_name}_R-{self.resize}_T-{self.default_tile_size}"


if __name__ == '__main__':
    device: torch.device = torch.device("cpu")
    configs = MosaicConfig()
    mosaic_maker = Blasi05MosaicMaker(configs)
    mosaic_maker.make(os.path.join("DiBlasi2005_outputs", configs.get_str()))

