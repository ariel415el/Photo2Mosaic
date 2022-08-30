import os.path
import argparse

from scipy.ndimage import binary_dilation

from utils import *
from utils.edge_detection import get_edges_map
from utils.common import create_direction_field
from utils.tesselation.oriented_tesselation import oriented_tesselation_with_edge_avoidance

ROTATION_MATRICES={x: get_rotation_matrix(x) for x in [45, 90, 135, 225, 315]}


def get_avoidance_map(edges_map, dilation_iterations=0):
    if dilation_iterations > 0:
        edges_map = binary_dilation(edges_map, iterations=1).astype(np.uint8)

    return edges_map

class HausnerMosaicMaker:
    @staticmethod
    def _design_mosaic(img, density_map, config, debug_dir):
        edge_map = get_edges_map(config.edges_reference, img)

        avoidance_map = get_avoidance_map(edge_map, config.edge_avoidance_dilation)

        direction_field, _ = create_direction_field(edge_map)

        centers, _ = oriented_tesselation_with_edge_avoidance(avoidance_map, direction_field, density_map, config.n_tiles, config.n_iters, debug_dir=debug_dir)
        oritentations = direction_field[centers[:, 0], centers[:, 1]]

        # Debug code:
        plot_vector_field(direction_field, img, path=os.path.join(debug_dir, 'VectorField.png'))
        cv2.imwrite(os.path.join(debug_dir, 'EdgeMap.png'), avoidance_map * 255)

        return centers, oritentations

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
    def make_mosaic(config, outputs_dir):
        debug_dir = os.path.join(outputs_dir, "debug")
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)

        image, density_map = load_images(config.img_path, config.density_map_path, config.output_height)

        h, w, _ = image.shape
        default_tile_size = int(config.delta * np.sqrt(h * w / config.n_tiles))

        centers, oritentations = HausnerMosaicMaker._design_mosaic(image, density_map, config, debug_dir)

        # write final mosaic
        HausnerMosaicMaker._render_tiles(image, density_map, centers, oritentations, default_tile_size, path=os.path.join(outputs_dir, f'FinalMosaic.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulating Decorative Mosaics by Hausner 2001.')
    parser.add_argument('img_path')
    parser.add_argument('--edges_reference', default='auto', help='Infer edge map automaticaly or from a binary mask file (Specify a file path)')
    parser.add_argument('--density_map_path', default=None, help='A binary mask specifying areas to tile with smaller tiles')
    parser.add_argument('--output_height', default=512)
    parser.add_argument('--n_tiles', default=1024)
    parser.add_argument('--n_iters', default=20, help='Number of iterations for the tesselation algorithm')
    parser.add_argument('--init_mode', default='random', help='Initialization mode of the tesseslation algorithm', choices=['random', 'uniform'])
    parser.add_argument('--delta', default=0.85, help='Factors the relative size of the tiles')
    parser.add_argument('--edge_avoidance_dilation', default=2, help='Stricter restriction on puting tiles over edges')

    configs = parser.parse_args()
    im_name = os.path.basename(os.path.splitext(configs.img_path)[0])
    configs.name = f"{im_name}_R-{configs.output_height}_N-{configs.n_tiles}_dlta-{configs.delta}"

    mosaic_maker = HausnerMosaicMaker.make_mosaic(configs, os.path.join("outputs", "Hausner2001", configs.name))


