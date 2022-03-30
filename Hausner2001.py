import os.path
from dataclasses import dataclass

import torch
from scipy import ndimage
from scipy.ndimage import binary_dilation

from utils import *
from utils.tesselation import VornoiTessealtion

ROTATION_MATRICES={x: get_rotation_matrix(x) for x in [45, 90, 135, 225, 315]}


def get_avoidance_map(edges_map, dilation_iterations=0):
    if dilation_iterations > 0:
        edges_map = binary_dilation(edges_map, iterations=1).astype(np.uint8)
    # mask = binary_erosion(mask, iterations=1).astype(np.uint8)

    return edges_map

def create_direction_field(edge_map):
    """
    Create a direction vector field from edges in the edges_map.
    Compute edges map its distance transform and then its gradient map.
    Normalize gradient vector to have unit norm.
    """
    dist_transform = ndimage.distance_transform_edt(edge_map == 0)
    dist_transform = cv2.GaussianBlur(dist_transform, (5, 5), 0)
    direction_field = np.stack(np.gradient(dist_transform), axis=2)
    direction_field = cv2.GaussianBlur(direction_field, (5, 5), 0)
    direction_field = normalize_vector_field(direction_field)

    return direction_field


class HausnerMosaicMaker:
    def __init__(self, configs):
        self.config = configs

        self.image, self.alpha_map = load_images(self.config.img_path, self.config.alpha_mask_path, self.config.resize)

        self.h, self.w, _ = self.image.shape
        self.default_tile_size = int(self.config.delta * np.sqrt(self.h * self.w / self.config.n_tiles))

    def make(self, outputs_dir):
        os.makedirs(outputs_dir, exist_ok=True)
        debug_dir = os.path.join(outputs_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        edge_map = get_edges_map_canny(self.image)

        avoidance_map = get_avoidance_map(edge_map, self.config.edge_avoidance_dilation)

        direction_field = create_direction_field(edge_map)

        vornoi_maker = VornoiTessealtion(self.config.n_tiles, self.alpha_map, self.config.initial_location, torch_device=device)
        centers, oritentations, _ = vornoi_maker.tesselate(direction_field, avoidance_map, self.config.n_iters, debug_dir=debug_dir)

        # write final mosaic
        self._render_tiles(centers, oritentations, path=os.path.join(outputs_dir, f'FinalMosaic.png'))

        # Debug code:
        plot_vector_field(direction_field, self.image, path=os.path.join(debug_dir, 'VectorField.png'))
        cv2.imwrite(os.path.join(debug_dir, 'EdgeMap.png'), avoidance_map * 255)

    def _render_tiles(self, centers, oritentations, path='mosaic.png'):
        """
        Render the mosaic: place oritented squares on a canvas with size defined by the alpha mask
        """
        mosaic = np.ones_like(self.image) * 127
        coverage_map = np.zeros(self.image.shape[:2])

        assert np.allclose(np.linalg.norm(oritentations,axis=1), 1)

        corner_directions = (oritentations @ np.stack([ROTATION_MATRICES[45],
                                                       ROTATION_MATRICES[135],
                                                       ROTATION_MATRICES[225],
                                                       ROTATION_MATRICES[315]]))

        corner_diameter = self.default_tile_size / np.sqrt(2)
        for i in range(centers.shape[0]):
            alpha = self.alpha_map[centers[i][0], centers[i][1]]

            contours = centers[i] + corner_directions[:, i] * corner_diameter / alpha
            contours = contours[:, ::-1]

            color = self.image[int(centers[i][0]), int(centers[i][1])]
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

@dataclass
class MosaicConfig:
    img_path: str = 'images/images/YingYang.png'
    alpha_mask_path: str = 'images/masks/YingYang_mask.png'
    resize: int = 512
    n_tiles: int = 1500
    n_iters: int = 10
    delta: float = 0.99  # Direction field variation level
    edge_avoidance_dilation: int = 2
    initial_location: str = 'random' # random / uniform
    debug_freq: int = 50

    def get_str(self):
        im_name = os.path.basename(os.path.splitext(self.img_path)[0])
        return f"{im_name}_R-{self.resize}_N-{self.n_tiles}_dlta-{self.delta}"


if __name__ == '__main__':
    device: torch.device = torch.device("cpu")
    configs = MosaicConfig()
    mosaic_maker = HausnerMosaicMaker(configs)
    mosaic_maker.make(os.path.join("outputs", "Hausner2001", configs.get_str()))

