import os
import random
from dataclasses import dataclass

from scipy.ndimage import binary_dilation, binary_closing, binary_opening
from skimage.morphology import skeletonize
from tqdm import tqdm

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.ndimage.measurements import label

import utils

FREE_SPACE_LABEL = 2
ROTATION_MATRICES={x: utils.get_rotation_matrix(x) for x in [45, 90]}


class MosaicDesigner:
    """Designs the level lines map and the direction field that is later used for tiling the image according to DiBlasi2005"""
    def __init__(self, default_tile_size, edges_reference, aligned_background, extra_level_gap, dist_transform_blur_params):
        self.default_tile_size = default_tile_size
        self.edges_reference = edges_reference
        self.aligned_background = aligned_background
        self.extra_level_gap = extra_level_gap
        self.dist_transform_blur_params = dist_transform_blur_params

    def get_edges_map(self, image, size_map):
        """Get an edge map in 3 ways:
         1: From the image itself using an edge detector
         2: Use the contour of the mask (size_map) as edges
         3: Use a predefined edge image. Possibly refine it to its skeleton
         """
        if self.edges_reference == 'mask' and size_map is not None:
            edge_map = utils.get_edges_map_canny(cv2.cvtColor(size_map * 127, cv2.COLOR_GRAY2BGR))
        elif os.path.exists(self.edges_reference):
            edge_map = cv2.imread(self.edges_reference, cv2.IMREAD_GRAYSCALE)
            edge_map = utils.set_image_height_without_distortion(edge_map, image.shape[0], mode=cv2.INTER_NEAREST)
            edge_map[edge_map < 127] = 0
            edge_map[edge_map >= 127] = 1
            edge_map = skeletonize(edge_map).astype(np.uint8)
        else:
            edge_map = utils.get_edges_map_canny(image, blur_size=7, sigma=5, t1=100, t2=150)

        return edge_map

    def get_level_matrix_from_dist_map(self, dist_map, size_map, offset=0.5):
        """
        Create binary mask where pixels in periodic distances from edges are turned on.
        Level lines frequency is determined by the size map.
        offset: float: 0.0 - 1.0 where to put the level maps
        """
        dist_map = dist_map.astype(int)
        level_matrix = np.zeros_like(dist_map).astype(np.uint8)
        default_levels_gap = self.default_tile_size + self.extra_level_gap
        for factor in np.unique(size_map):
            gap_size = default_levels_gap // factor
            mask = np.remainder(dist_map, gap_size) == int(offset * gap_size)
            mask = np.logical_and(mask, size_map == factor)
            level_matrix[mask] = FREE_SPACE_LABEL
        return level_matrix

    def create_gradient_and_level_matrix(self, edge_map, size_map):
        """
        Compute distance transform from edges, return normalized gradient of distances
        Compute a level map from integer modolu of distances
        """
        dist_transform = ndimage.distance_transform_edt(edge_map == 0)

        if self.dist_transform_blur_params is not None:
            k, s = self.dist_transform_blur_params
            dist_transform = cv2.GaussianBlur(dist_transform, (k, k), s)
        level_matrix = self.get_level_matrix_from_dist_map(dist_transform, size_map, offset=0.5)

        # level_matrix += skeletonize(edge_map).astype(np.uint8) * FREE_SPACE_LABEL add the edges themselves as a leveline

        direction_field = np.stack(np.gradient(dist_transform), axis=2)
        direction_field = utils.normalize_vector_field(direction_field)

        # Use default background
        if self.aligned_background:
            if np.all(size_map):
                raise UserWarning("aligned_background should be used only with a non-empty mask")
            # Create separate directions and levels for background
            dilated_foreground = binary_dilation(size_map != 1, iterations=self.default_tile_size * 4)
            bg_level_matrix = np.zeros_like(level_matrix)

            # Set BG level lines to horizontal lines
            gap = self.default_tile_size + self.extra_level_gap
            bg_level_matrix[::gap] = FREE_SPACE_LABEL
            level_matrix = np.where(dilated_foreground, level_matrix, bg_level_matrix)

            # Set Bg orientatins to [1,0]
            bg_direction_field = np.ones_like(direction_field)
            bg_direction_field[..., 0] = 0
            direction_field = np.where(dilated_foreground[..., None], direction_field, bg_direction_field)

        return direction_field, level_matrix

    def get_design(self, image, size_map, debug_dir):
        edge_map = self.get_edges_map(image, size_map)
        direction_field, level_matrix = self.create_gradient_and_level_matrix(edge_map, size_map)

        # Dumb debug images
        utils.plot_vector_field(direction_field, edge_map * 255, path=os.path.join(debug_dir, 'VectorField.png'))
        cv2.imwrite(os.path.join(debug_dir, 'size_map.png'), size_map * 127)
        cv2.imwrite(os.path.join(debug_dir, 'EdgeMap.png'), edge_map * 255)
        cv2.imwrite(os.path.join(debug_dir, 'Level_matrix.png'), level_matrix * 127)

        return direction_field, level_matrix


class MosaicTiler:
    def __init__(self, default_tile_size, delete_area_factor, cement_color, aspect_ratio, debug_freq):
        self.default_tile_size = default_tile_size
        self.delete_area_factor = delete_area_factor
        self.cement_color = cement_color
        self.aspect_ratio = aspect_ratio
        self.debug_freq = debug_freq

    def find_approximate_location(self, center, normal, diameter, candidate_location_map):
        """
        Look in the orientedd surrounding of 'center' for free space in 'candidate_location_map'
        """
        # arange = np.arange(diameter // 2, diameter)
        # arange = np.concatenate([-arange, arange])
        arange = np.arange(-diameter, diameter)
        rs, cs = np.meshgrid(arange, arange)
        all_offsets = np.column_stack((rs.ravel(), cs.ravel()))
        basis2 = normal @ ROTATION_MATRICES[90] * self.aspect_ratio

        offset_dist_func = lambda x: np.abs((x * normal).sum(-1)) + np.abs((x * basis2).sum(-1))
        all_offsets = sorted(list(all_offsets), key=offset_dist_func)

        for offset in all_offsets:
            point = center + offset
            point = np.minimum(point, np.array(candidate_location_map.shape) - 1)
            point = np.maximum(point, np.array([0, 0])).astype(int)
            if candidate_location_map[point[0], point[1]] == FREE_SPACE_LABEL:
                return point
        return None

    @staticmethod
    def get_tile_rectangle_contour(center, tile_size, normal, aspect_ratio):
        """Get a cv2 drawable contour of a rectangle centered in 'center' with 2 faces in normal1, normal2"""
        vec1 = normal
        vec2 = normal @ ROTATION_MATRICES[90] * aspect_ratio
        corner_directions = np.array([vec1 + vec2,
                                      vec1 - vec2,
                                      -vec1 - vec2,
                                      -vec1 + vec2])

        contours = center + corner_directions * (tile_size / 2)
        contours = contours[:, ::-1]
        box = np.int0(cv2.boxPoints(cv2.minAreaRect(np.array(contours).astype(int))))

        return box

    @staticmethod
    def get_track_maps_from_level_map(level_map):
        """
        separate the different levellines with 2d conencted components so that each track can be approached separately
        """
        structure = np.ones((3, 3), dtype=np.int)
        all_tracks, ncomponents = label(level_map, structure)

        track_maps = [(all_tracks == i).astype(np.uint8) * FREE_SPACE_LABEL for i in range(1, ncomponents + 1)]

        return track_maps

    def get_next_tile_position(self, position, direction_field, size_map, candidate_location_map):
        if position is not None:  # Find approximate location
            search_diameter = self.delete_area_factor * (self.default_tile_size // size_map[position[0], position[1]]) * 2
            normal = direction_field[position[0], position[1]]
            position = self.find_approximate_location(position, normal, search_diameter, candidate_location_map)

        if position is None:  # Find loose track ends
            num_neighbors = convolve2d(candidate_location_map == FREE_SPACE_LABEL, np.ones((3, 3)), mode='same', fillvalue=-9)
            num_neighbors *= (candidate_location_map == FREE_SPACE_LABEL)
            values = np.unique(num_neighbors)
            min_value = values[values != 0].min()
            nwhere = np.where(num_neighbors == min_value)
            idx = np.random.randint(0, len(nwhere[0]))
            position = np.array([nwhere[0][idx], nwhere[1][idx]])

        # if position is None:  # find a random free-space
        #     candidate_points = np.stack(np.where(candidate_location_map == FREE_SPACE_LABEL), axis=1)
        #     position = candidate_points[np.random.randint(0, len(candidate_points))]

        return position

    def render_tiles(self, image, size_map, direction_field, level_matrix, debug_dir):
        """
        Render a mosaic: place oritented squares on a canvas with size defined by the a mask
        :param: image: Used to get the colors of the tiles
        :param: size_map: defines the size of the tiles placed in each location
        :param: image: Dictates the orientations of the placed tiles
        :param: image: A binary mask that instructs where to put tiles
        """
        mosaic = np.ones_like(image) * self.cement_color
        coverage_map = np.zeros(image.shape[:2])

        # Use quantized colors
        color_map = cv2.GaussianBlur(image, (7, 7), 0)
        color_map = (color_map // 32) * 32

        all_track_maps = MosaicTiler.get_track_maps_from_level_map(level_matrix)

        pbar = tqdm()
        n_placed_tiles = 0
        for track_map in all_track_maps[3:]:
            center = None
            while (track_map == FREE_SPACE_LABEL).any():

                center = self.get_next_tile_position(center, direction_field, size_map, track_map)

                # Get tile details
                normal = direction_field[center[0], center[1]]
                tile_color = color_map[center[0], center[1]].tolist()
                local_tile_size = self.default_tile_size / size_map[center[0], center[1]]
                contour = MosaicTiler.get_tile_rectangle_contour(center, local_tile_size, normal, self.aspect_ratio)

                # Draw oriented tile
                cv2.drawContours(mosaic, [contour], -1, color=tile_color, thickness=cv2.FILLED)
                cv2.drawContours(mosaic, [contour], -1, color=(0, 0, 0), thickness=1)

                # Update coverage_map
                tmp = np.zeros_like(coverage_map)
                cv2.drawContours(tmp, [contour], -1, color=1, thickness=cv2.FILLED)
                coverage_map += tmp

                # Remove surrounding from candidate points
                delete_area = local_tile_size * self.delete_area_factor
                contour = MosaicTiler.get_tile_rectangle_contour(center, delete_area, normal, self.aspect_ratio)
                cv2.drawContours(track_map, [contour], -1, color=0, thickness=cv2.FILLED)

                n_placed_tiles += 1
                pbar.set_description(f"Placed tiles: {n_placed_tiles}")

                if n_placed_tiles % self.debug_freq == 0:
                    tmp = mosaic + ((track_map == 2).astype(np.uint8) * 255 * 0.4)[:, :, None]
                    # Debug Draw tile normal
                    center_xy = center[::-1].astype(int)
                    direction_xy = (normal[::-1] * self.default_tile_size / 2).astype(int)
                    cv2.arrowedLine(tmp, tuple(center_xy), tuple(center_xy + direction_xy), (0, 0, 255), 1)
                    cv2.imwrite(os.path.join(debug_dir, f"Moisaic-step-{n_placed_tiles}.png"), tmp)

        coverage_percentage = (coverage_map > 0).sum() / coverage_map.size * 100
        overlap_percentage = (coverage_map > 1).sum() / coverage_map.size * 100
        print(f"Coverage: {coverage_percentage:.1f}% Overlap:{overlap_percentage:.1f}%)")
        return mosaic, coverage_percentage, overlap_percentage


def make_mosaic(config, outputs_dir):
    image, size_map = utils.load_images(*config.get_image_configs())
    utils.create_output_dirs(outputs_dir)

    designer = MosaicDesigner(*config.get_design_configs())
    direction_field, level_matrix = designer.get_design(image, size_map, os.path.join(outputs_dir, "debug"))

    tiler = MosaicTiler(*config.get_tiling_configs())
    mosaic, coverage_percentage, overlap_percentage = tiler.render_tiles(image, size_map, direction_field, level_matrix, debug_dir=os.path.join(outputs_dir, f'debug'))

    cv2.imwrite(os.path.join(outputs_dir, 'FinalMosaic.png'), mosaic)


@dataclass
class MosaicConfig:
    # io
    img_path: str = 'images/images/Alexander_body.jpg'
    size_map_path: str = None
    resize: int = 1024

    # Common
    default_tile_size = 10

    # Design
    edges_reference: str = 'image'   # path/image/mask compute edges from image itself or from the mask
    aligned_background = False  # Reauires mask. mask == 2 is the foreground
    extra_level_gap = 0  # the gap between level lines will be tile_size + levels_gap
    dist_transform_blur_params = (5, 1)
    
    # Tiling
    delete_area_factor:float = 2.  # determines the size of the minimal gap between placed tiles (multiplies the tile diameter)
    cement_color: int = 127
    aspect_ratio: float = 2

    # debug
    debug_freq = 100

    def get_image_configs(self):
        return self.img_path, self.size_map_path, self.resize

    def get_design_configs(self):
        return self.default_tile_size, self.edges_reference, self.aligned_background, self.extra_level_gap, self.dist_transform_blur_params

    def get_tiling_configs(self):
        return self.default_tile_size, self.delete_area_factor, self.cement_color, self.aspect_ratio, self.debug_freq

    def get_str(self):
        im_name = os.path.basename(os.path.splitext(self.img_path)[0])
        name = f"{im_name}_R-{self.resize}_T-{self.default_tile_size}_D-{self.delete_area_factor}"
        name += f"ER-{'EdgeMap' if os.path.exists(self.edges_reference) else self.edges_reference}"
        name += f"SM" if self.size_map_path is not None else ''
        name += f"AB" if self.aligned_background else ''

        return name

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    mosaic_configs = MosaicConfig()
    make_mosaic(mosaic_configs, os.path.join("outputs", "DiBlasi2005", mosaic_configs.get_str()))
