import os.path
from dataclasses import dataclass

import numpy as np
from scipy.ndimage.measurements import label

from utils import *
from utils import get_edges_map_canny
from utils.tesselation.oriented_tesselation import oriented_tesselation_with_edge_avoidance

ROTATION_MATRICES={x: get_rotation_matrix(x) for x in [45, 90, 135, 225, 315]}

def get_level_lines(edges_map, density_map, tile_size):
    level_matrix = edges_map.copy()
    direction_field, dist_transform = create_direction_field(edges_map)
    dist_map = dist_transform.astype(int)

    for factor in np.unique(density_map):
        gap_size = tile_size // factor
        remainder = np.remainder(dist_map, gap_size)
        # mask = (remainder == 0) | (remainder == 1) | (remainder == gap_size -1)
        mask = (remainder == 0)
        mask = mask & (density_map == factor)
        level_matrix[mask] = 1

    # level_matrix = binary_dilation(level_matrix, iterations=1).astype(np.uint8)

    return level_matrix

def get_track_maps_from_level_map(level_map):
    """
    separate the different levellines with 2d conencted components so that each track can be approached separately
    """
    structure = np.ones((3, 3), dtype=np.int)
    all_tracks, ncomponents = label(level_map, structure)

    track_maps = [(all_tracks == i).astype(np.uint8) for i in range(1, ncomponents + 1)]

    return track_maps

def get_centers(level_lines_map, tile_size):
    all_track_maps = get_track_maps_from_level_map(level_lines_map)

def get_n_tiles(tile_size, density_map, delta):
    n_tiles = 0
    for v in np.unique(density_map):
        area = np.sum(density_map==v)
        n_tiles += area /tile_size**2

    return int(n_tiles / delta)

class MyMosaicMaker:
    @staticmethod
    def _design_mosaic(img, density_map, config, debug_dir):
        edge_map = get_edges_map_canny(img, t1=50, t2=100)

        direction_field, _ = create_direction_field(edge_map)
        level_lines_map = get_level_lines(edge_map, density_map, config.tile_size)

        n_tiles = get_n_tiles(config.tile_size, density_map, config.delta)
        print(n_tiles)

        centers, label_map = oriented_tesselation_with_edge_avoidance(level_lines_map, direction_field, density_map,
                                                              n_tiles, config.n_iters, cut_by_edges=True, debug_dir=debug_dir)
        oritentations = direction_field[centers[:, 0], centers[:, 1]]

        # Debug code:
        plot_vector_field(direction_field, img, path=os.path.join(debug_dir, 'VectorField.png'))
        cv2.imwrite(os.path.join(debug_dir, 'Avodiance_map.png'), level_lines_map * 255)
        cv2.imwrite(os.path.join(debug_dir, 'Edges.png'), edge_map * 255)

        return centers, oritentations, label_map

    @staticmethod
    def _render_tiles(img, density_map, centers, oritentations, default_tile_size, path='mosaic.png'):
        """
        Render the mosaic: place oritented squares on a canvas with size defined by the alpha mask
        """
        mosaic = np.ones_like(img) * 127
        coverage_map = np.zeros(img.shape[:2])

        assert np.allclose(np.linalg.norm(oritentations,axis=1), 1)

        corner_directions = (oritentations @ np.stack([ROTATION_MATRICES[45],
                                                       ROTATION_MATRICES[135],
                                                       ROTATION_MATRICES[225],
                                                       ROTATION_MATRICES[315]]))

        corner_diameter = default_tile_size / np.sqrt(2)
        for i in range(centers.shape[0]):
            alpha = density_map[centers[i][0], centers[i][1]]

            contours = centers[i] + corner_directions[:, i] * corner_diameter / alpha
            contours = contours[:, ::-1]

            color = img[int(centers[i][0]), int(centers[i][1])]
            box = np.int0(cv2.boxPoints(cv2.minAreaRect(np.array(contours).astype(int))))
            cv2.drawContours(mosaic, [box], -1, color=color.tolist(), thickness=cv2.FILLED)
            # cv2.drawContours(mosaic, [box], -1, color=(0,0,0), thickness=1)

            # update overidden pixels
            tmp = np.zeros_like(coverage_map)
            cv2.drawContours(tmp, [box], -1, color=1, thickness=cv2.FILLED)
            coverage_map += tmp

        coverage_percentage = (coverage_map > 0).sum() / coverage_map.size * 100
        overlap_percentage = (coverage_map > 1).sum() / coverage_map.size * 100
        print(f"Coverage: {coverage_percentage:.3f}% Overlap:{overlap_percentage:.3f}%)")
        cv2.imwrite(path, mosaic)

        return coverage_percentage, overlap_percentage

    @staticmethod
    def _render_cells(img, label_map, path='mosaic.png'):
        mosaic = np.zeros_like(img)
        for label in np.unique(label_map):
            mosaic[label_map == label] = np.mean(img[label_map == label], axis=0)

        cv2.imwrite(path, mosaic)

    @staticmethod
    def make_mosaic(config, outputs_dir):
        debug_dir = os.path.join(outputs_dir, "debug")
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)

        image, density_map = load_images(config.img_path, config.alpha_mask_path, config.resize)

        centers, oritentations, label_map = MyMosaicMaker._design_mosaic(image, density_map, config, debug_dir)

        # write final mosaic
        # MyMosaicMaker._render_tiles(image, density_map, centers, oritentations, config.tile_size, path=os.path.join(outputs_dir, f'FinalMosaic.png'))
        MyMosaicMaker._render_cells(image, label_map, path=os.path.join(outputs_dir, f'FinalMosaic.png'))


@dataclass
class MosaicConfig:
    img_path: str = 'images/images/turk.jpg'
    alpha_mask_path: str = 'images/masks/turk_mask.png'

    resize: int = 512
    tile_size: int = 30
    n_iters: int = 10
    delta: float = 0.6 # Direction field variation level
    edge_avoidance_dilation: int = 2
    init_mode: str = 'random' # random / uniform
    debug_freq: int = 50

    def get_str(self):
        im_name = os.path.basename(os.path.splitext(self.img_path)[0])
        return f"{im_name}_R-{self.resize}_T-{self.tile_size}_dlta-{self.delta}"


if __name__ == '__main__':
    configs = MosaicConfig()
    mosaic_maker = MyMosaicMaker.make_mosaic(configs, os.path.join("outputs", "my_method", configs.get_str()))

