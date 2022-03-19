import os.path
from dataclasses import dataclass

import torch
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm

from utils import *

ROTATION_MATRICES={x: get_rotation_matrix(x) for x in [45, 90, 135, 225, 315]}


def get_avoidance_map(edges_map, dilation_iterations=0):
    if dilation_iterations > 0:
        edges_map = binary_dilation(edges_map, iterations=1).astype(np.uint8)
    # mask = binary_erosion(mask, iterations=1).astype(np.uint8)

    return edges_map


def update_point_locations(points, oritentations, direction_field, avoidance_map, alpha_map):
    """
    Iterative approach for computing centridal Vornoi cells using brute force equivalet of the Z-buffer algorithm
    For point in "points" compute a distance map now argmin over all maps to get a single map with index of
    closest point to each coordinate.
    The metric is used is the L1 distance with axis rotated by the points' orientation
    """

    h, w = direction_field.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([yy, xx], axis=-1)  # coords[a ,b] = [b,a]

    diffs = coords - points[:, None, None]

    basis1 = oritentations[:, None, None]
    basis2 = (oritentations @ ROTATION_MATRICES[90])[:, None, None]

    # Heavy computation on GPU
    with torch.no_grad():
        gaps = torch.from_numpy(diffs).to(device)
        basis1 = torch.from_numpy(basis1).to(device)  # u
        basis2 = torch.from_numpy(basis2).to(device)  # v

        # L1_u-v((x,y),(x0,y0)) = |<(x,y),u> - <(x0,y0),u>| + |<(x,y),v> - <(x0,y0),v>| = |<(x,y)-(x0,y0),u>| + |<(x,y)-(x0,y0),u>|
        distance_maps = torch.abs((gaps * basis1).sum(-1)) + torch.abs((gaps * basis2).sum(-1))
        distance_maps = distance_maps.cpu().numpy()

    for v in np.unique(alpha_map):
        distance_maps[alpha_map[points[:, 0], points[:, 1]] == v] *= v

    vornoi_map = np.argmin(distance_maps, axis=0)

    # Mark edges as ex-teritory (push cells away from edges)
    vornoi_map[avoidance_map == 1] = vornoi_map.max() + 2

    # ensure non-empty cells (points may not be closest to itself since argmax between same values is arbitraray)
    vornoi_map[points[:, 0], points[:, 1]] = np.arange(len(points))

    # Move points to be the centers of their according vornoi cells
    centers = np.array([coords[vornoi_map==i].mean(0) for i in range(len(points))]).astype(np.uint64)

    # update orientations using vector field
    oritentations = direction_field[centers[:, 0].astype(int), centers[:, 1].astype(int)]

    return centers, oritentations, vornoi_map


def create_direction_field(edge_map):
    """
    Create a direction vector field from edges in the edges_map.
    Compute edges map its distance transform and then its gradient map.
    Normalize gradient vector to have unit norm.
    """
    dist_transform = ndimage.distance_transform_edt(edge_map == 0)
    dist_transform = cv2.GaussianBlur(dist_transform, (5, 5), 2)
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

        points, oritentations = self._get_initial_state()

        coverage = []
        overlap = []
        for i in tqdm(range(self.config.n_iters)):
            points, oritentations, debug_vornoi_diagram = update_point_locations(points, oritentations, direction_field, avoidance_map, self.alpha_map)

            if i % self.config.debug_freq == 0:
                plot_vornoi_cells(debug_vornoi_diagram, points, oritentations, avoidance_map, path=os.path.join(debug_dir, f'Vornoi_diagram_{i}.png'))
                coverage_percentage, overlap_percentage = self._render_tiles(points, oritentations, path=os.path.join(debug_dir, f'Mosaic_{i}.png'))
                coverage.append(coverage_percentage)
                overlap.append(overlap_percentage)

                plt.plot(np.arange(len(coverage)), coverage, label='Coverage')
                plt.plot(np.arange(len(overlap)), overlap, label='Overlap')
                plt.legend()
                plt.savefig(os.path.join(debug_dir, 'Loss.png'))
                plt.clf()

        # write final mosaic
        self._render_tiles(points, oritentations, path=os.path.join(outputs_dir, f'FinalMosaic.png'))

        # Debug code:
        plot_vector_field(direction_field, self.image, path=os.path.join(debug_dir, 'VectorField.png'))
        cv2.imwrite(os.path.join(debug_dir, 'EdgeMap.png'), avoidance_map * 255)

    def _get_initial_state(self):
        """Sample initial tile centers and orientations  cell centers and directions"""
        N = self.config.n_tiles
        h, w = self.h, self.w
        if self.config.initial_location == 'random': # sample randomly in each are in proportion to tile sizes in it
            dense_area_proportion = (self.alpha_map == 2).sum() / self.alpha_map.size

            # if X is the num_dense_area_points then the sparse area has X/4 and solving leads to
            num_dense_area_points = int(4 * dense_area_proportion * N / (1 + 4 * dense_area_proportion))
            num_default_area_points = N - num_dense_area_points

            posible_points = np.stack(np.where(self.alpha_map == 2), axis=1)
            dense_positions = posible_points[np.random.randint(len(posible_points), size=num_dense_area_points)]
            posible_points = np.stack(np.where(self.alpha_map == 1), axis=1)
            default_positions = posible_points[np.random.randint(len(posible_points), size=num_default_area_points)]
            points = np.concatenate([default_positions, dense_positions], axis=0)

        else: # uniform
            s = int(np.ceil(np.sqrt(N)))
            y, x = np.meshgrid(np.linspace(1, h - 1, s), np.linspace(1, w - 1, s))
            points = np.stack([y.flatten(), x.flatten()], axis=1).astype(np.uint64)[:N]

        oritentations = np.random.rand(N, 2)
        oritentations /= np.linalg.norm(oritentations, axis=1, keepdims=True)

        return points, oritentations

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
    img_path: str = 'images/YingYang.png'
    alpha_mask_path: str = 'images/YingYang_mask.png'
    resize: int = 256
    n_tiles: int = 500
    n_iters: int = 50
    delta: float = 1.2  # Direction field variation level
    edge_avoidance_dilation: int = 2
    initial_location: str = 'random' # random / uniform
    debug_freq: int = 50

    def get_str(self):
        im_name = os.path.basename(os.path.splitext(self.img_path)[0])
        return f"{im_name}_R-{self.resize}_N-{self.n_tiles}_dlta-{self.delta}"


if __name__ == '__main__':
    device: torch.device = torch.device("cuda:0")
    configs = MosaicConfig()
    mosaic_maker = HausnerMosaicMaker(configs)
    mosaic_maker.make(os.path.join("Hausner2001_outputs", configs.get_str()))

