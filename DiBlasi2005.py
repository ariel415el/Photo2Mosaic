import argparse
import os
import random
from dataclasses import dataclass

from scipy.ndimage import binary_dilation, binary_closing, binary_opening
from skimage.morphology import skeletonize
from tqdm import tqdm

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.ndimage.measurements import label

import utils

import utils.common
import utils.edge_detection

FREE_SPACE_LABEL = 2
ROTATION_MATRICES={x: utils.get_rotation_matrix(x) for x in [-45, 45, 90]}


class MosaicDesigner:
    """Designs the level lines map and the direction field that is later used for tiling the image according to DiBlasi2005"""
    def __init__(self, default_tile_size, edges_reference, aligned_background, extra_level_gap):
        self.default_tile_size = default_tile_size
        self.edges_reference = edges_reference
        self.aligned_background = aligned_background
        self.extra_level_gap = extra_level_gap

    def create_gradient_and_level_matrix(self, edge_map, density_map):
        """
        Compute distance transform from edges, return normalized gradient of distances
        Compute a level map from integer modolu of distances
        """

        default_levels_gap = self.default_tile_size + self.extra_level_gap
        level_matrix, direction_field, dist_transform = utils.get_level_lines(edge_map, density_map, default_levels_gap, offset=0.5)
        level_matrix[level_matrix == 1] = FREE_SPACE_LABEL

        # Use default background
        if self.aligned_background:
            if np.all(density_map == 1):
                raise UserWarning("aligned_background should be used only with a non-empty mask")
            # Create separate directions and levels for background
            dilated_foreground = binary_dilation(density_map != 1, iterations=self.default_tile_size * 1)
            bg_level_matrix = np.zeros_like(level_matrix)

            # Set BG level lines to horizontal lines
            gap = self.default_tile_size + self.extra_level_gap
            bg_level_matrix[::gap] = FREE_SPACE_LABEL
            level_matrix = np.where(dilated_foreground, level_matrix, bg_level_matrix)

            # Set Bg orientatins to [1,0]
            bg_direction_field = np.ones_like(direction_field)
            bg_direction_field[..., 0] = 0
            direction_field = np.where(dilated_foreground[..., None], direction_field, bg_direction_field)

        # Dumb debug images
        utils.plot_vector_field(direction_field, edge_map * 255, path=os.path.join(self.debug_dir, 'VectorField.png'))
        cv2.imwrite(os.path.join(self.debug_dir, 'density_map.png'), density_map * 127)
        cv2.imwrite(os.path.join(self.debug_dir, 'Level_matrix.png'), level_matrix * 127)
        cv2.imwrite(os.path.join(self.debug_dir, 'dist_transform.png'), dist_transform / dist_transform.max() * 256)

        return direction_field, level_matrix

    def get_design(self, image, density_map, debug_dir):
        self.debug_dir = debug_dir

        edge_map = utils.edge_detection.get_edges_map(self.edges_reference, image, density_map)
        cv2.imwrite(os.path.join(self.debug_dir, 'EdgeMap.png'), edge_map * 255)

        direction_field, level_matrix = self.create_gradient_and_level_matrix(edge_map, density_map)

        return direction_field, level_matrix


class MosaicTiler:
    def __init__(self, default_tile_size, delete_area_factor, cement_color, aspect_ratio, debug_freq):
        self.default_tile_size = default_tile_size
        self.delete_area_factor = delete_area_factor
        self.cement_color = cement_color
        self.aspect_ratio = aspect_ratio
        self.debug_freq = debug_freq

    def find_approximate_location(self, center, direction, diameter, candidate_location_map):
        """
        Look in the orientedd surrounding of 'center' for free space in 'candidate_location_map'
        """
        arange = np.arange(-diameter, diameter)
        rs, cs = np.meshgrid(arange, arange)
        all_offsets = np.column_stack((rs.ravel(), cs.ravel()))
        basis1 = direction @ ROTATION_MATRICES[45]
        basis2 = direction @ ROTATION_MATRICES[-45] * self.aspect_ratio

        offset_dist_func = lambda x: np.abs((x * basis1).sum(-1)) + np.abs((x * basis2).sum(-1))
        all_offsets = sorted(list(all_offsets), key=offset_dist_func)

        for offset in all_offsets:
            point = center + offset
            point = np.minimum(point, np.array(candidate_location_map.shape) - 1)
            point = np.maximum(point, np.array([0, 0])).astype(int)
            if candidate_location_map[point[0], point[1]] == FREE_SPACE_LABEL:
                return point
        return None

    @staticmethod
    def get_tile_rectangle_contour(center, tile_size, direction, aspect_ratio):
        """Get a cv2 drawable contour of a rectangle centered in 'center' with 2 faces in normal1, normal2"""
        vec1 = direction
        vec2 = direction @ ROTATION_MATRICES[90] * aspect_ratio
        corner_directions = np.array([vec1 + vec2,
                                      vec1 - vec2,
                                      -vec1 - vec2,
                                      -vec1 + vec2])

        contours = center + corner_directions * (tile_size / 2)
        contours = contours[:, ::-1]
        box = np.int0(np.round(cv2.boxPoints(cv2.minAreaRect(np.array(contours).astype(int)))))

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

    def get_next_tile_position(self, position, direction_field, density_map, candidate_location_map):
        if position is not None:  # Find approximate location
            search_diameter = self.delete_area_factor * (self.default_tile_size // density_map[position[0], position[1]]) * 2
            direction = direction_field[position[0], position[1]]
            position = self.find_approximate_location(position, direction, search_diameter, candidate_location_map)

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

    def render_tiles(self, image, density_map, direction_field, level_matrix, debug_dir):
        """
        Render a mosaic: place oritented squares on a canvas with size defined by the a mask
        :param: image: Used to get the colors of the tiles
        :param: density_map: defines the size of the tiles placed in each location
        :param: image: Dictates the orientations of the placed tiles
        :param: image: A binary mask that instructs where to put tiles
        """
        mosaic = np.ones_like(image) * self.cement_color
        coverage_map = np.zeros(image.shape[:2])

        # Use quantized colors
        color_map = cv2.GaussianBlur(image, (7, 7), 0)
        # color_map = (color_map // 32) * 32

        all_track_maps = MosaicTiler.get_track_maps_from_level_map(level_matrix)

        pbar = tqdm()
        n_placed_tiles = 0
        for track_map in all_track_maps:
            center = None
            while (track_map == FREE_SPACE_LABEL).any():

                center = self.get_next_tile_position(center, direction_field, density_map, track_map)

                # Get tile details
                direction = direction_field[center[0], center[1]]
                tile_color = color_map[center[0], center[1]].tolist()
                local_tile_size = self.default_tile_size / density_map[center[0], center[1]]
                contour = MosaicTiler.get_tile_rectangle_contour(center, local_tile_size, direction, self.aspect_ratio)

                # Draw oriented tile
                cv2.drawContours(mosaic, [contour], -1, color=tile_color, thickness=cv2.FILLED)
                cv2.drawContours(mosaic, [contour], -1, color=(0, 0, 0), thickness=1)

                # Update coverage_map
                tmp = np.zeros_like(coverage_map)
                cv2.drawContours(tmp, [contour], -1, color=1, thickness=cv2.FILLED)
                coverage_map += tmp

                # Remove surrounding from candidate points
                delete_area = local_tile_size * self.delete_area_factor
                contour = MosaicTiler.get_tile_rectangle_contour(center, delete_area, direction, self.aspect_ratio)
                cv2.drawContours(track_map, [contour], -1, color=0, thickness=cv2.FILLED)

                n_placed_tiles += 1
                pbar.set_description(f"Placed tiles: {n_placed_tiles}")

                if n_placed_tiles % self.debug_freq == 0:
                    tmp = mosaic + ((track_map == 2).astype(np.uint8) * 255 * 0.4)[:, :, None]
                    # Debug Draw tile normal
                    center_xy = center[::-1].astype(int)
                    direction_xy = (direction[::-1] * self.default_tile_size / 2).astype(int)
                    cv2.arrowedLine(tmp, tuple(center_xy), tuple(center_xy + direction_xy), (0, 0, 255), 1)
                    cv2.imwrite(os.path.join(debug_dir, f"Moisaic-step-{n_placed_tiles}.png"), tmp)


        all_track_maps = MosaicTiler.get_track_maps_from_level_map(level_matrix)
        leftovers = np.sum(all_track_maps, axis=0) > 0
        leftovers *= (coverage_map > 0)
        cv2.imwrite(os.path.join(debug_dir, f"LeftOvers.png"), leftovers*255)

        coverage_percentage = (coverage_map > 0).sum() / coverage_map.size * 100
        overlap_percentage = (coverage_map > 1).sum() / coverage_map.size * 100
        print(f"Coverage: {coverage_percentage:.1f}% Overlap:{overlap_percentage:.1f}%)")
        return mosaic, coverage_percentage, overlap_percentage


def make_mosaic(config, outputs_dir):
    image, density_map = utils.load_images(config.img_path, config.density_map_path, config.output_height)
    utils.create_output_dirs(outputs_dir)

    designer = MosaicDesigner(*config.get_design_configs())
    direction_field, level_matrix = designer.get_design(image, density_map, os.path.join(outputs_dir, "debug"))

    tiler = MosaicTiler(*config.get_tiling_configs())
    mosaic, coverage_percentage, overlap_percentage = tiler.render_tiles(image, density_map, direction_field, level_matrix, debug_dir=os.path.join(outputs_dir, f'debug'))

    cv2.imwrite(os.path.join(outputs_dir, 'FinalMosaic.png'), mosaic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulating Decorative Mosaics by DiBlasi 2005.')
    # IO
    parser.add_argument('img_path')
    parser.add_argument('--edges_reference', default='auto', help='Infer edge map automaticaly or from a binary mask file (Specify a file path)')
    parser.add_argument('--density_map_path', default=None, help='A binary mask specifying areas to tile with smaller tiles')
    parser.add_argument('--output_height', default=512)

    # Design
    parser.add_argument('--default_tile_size', default=12, help='Width of the tile used for the non dense area')
    parser.add_argument('--aligned_background', action='store_true', default=False, help='Put aligned tiles in background. Reauires mask. mask == 2 is the foreground')
    parser.add_argument('--extra_level_gap', default=1, help='The gap between level lines will be tile_size + levels_gap')
    parser.add_argument('--delete_area_factor', default=2, help='Determines the size of the minimal gap between placed tiles (multiplies the tile diameter')
    parser.add_argument('--cement_color', default=127, help='Canvas color')
    parser.add_argument('--aspect_ratio', default=1, help='Defines the shape of the tiles')
    parser.add_argument('--debug_freq', default=100, help='How often to dump debug images')

    configs = parser.parse_args()
    configs.get_image_configs = lambda: (configs.img_path, configs.density_map_path, configs.output_height)
    configs.get_design_configs = lambda: (configs.default_tile_size, configs.edges_reference, configs.aligned_background, configs.extra_level_gap)
    configs.get_tiling_configs = lambda: (configs.default_tile_size, configs.delete_area_factor, configs.cement_color, configs.aspect_ratio, configs.debug_freq)

    im_name = os.path.basename(os.path.splitext(configs.img_path)[0])
    configs.name = f"{im_name}_R-{configs.output_height}_N-{configs.default_tile_size}_dlta-{configs.delete_area_factor}"
    configs.name += f"ER-{'EdgeMap' if os.path.exists(configs.edges_reference) else configs.edges_reference}"
    configs.name += f"SM" if configs.density_map_path is not None else ''
    configs.name += f"AB" if configs.aligned_background else ''

    make_mosaic(configs, os.path.join("outputs", "DiBlasi2005", configs.name))