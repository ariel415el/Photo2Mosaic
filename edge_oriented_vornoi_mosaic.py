import os.path
from dataclasses import dataclass

from utils import *
from utils import get_level_lines
from utils.edge_detection import get_edges_map
from utils.common import create_direction_field
from utils.tesselation.common import simplify_label_map, render_label_map_with_image
from utils.tesselation.oriented_tesselation import oriented_tesselation_with_edge_avoidance


def design_mosaic(img, density_map, config, debug_dir):
    edge_map = get_edges_map(config.edges_reference, img, density_map)

    direction_field, _ = create_direction_field(edge_map)

    if config.edges_mode == 'edges_mode':
        h, w = img.shape[:2]
        tile_size = int(np.sqrt(h * w / config.n_tiles))
        edge_map = get_level_lines(edge_map, density_map, tile_size)

    centers, label_map = oriented_tesselation_with_edge_avoidance(edge_map, direction_field, density_map,
                                                                  config.n_tiles, config.n_iters,
                                                                  search_area_factor=2, cut_by_edges=True,
                                                                  debug_dir=debug_dir)

    # Debug code:
    plot_vector_field(direction_field, img, path=os.path.join(debug_dir, 'VectorField.png'))
    cv2.imwrite(os.path.join(debug_dir, 'EdgeMap.png'), edge_map * 255)

    return centers, label_map


def make_edge_oriented_vornoi_mosaic(config, outputs_dir):
    debug_dir = os.path.join(outputs_dir, "debug")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    image, density_map = load_images(config.img_path, config.alpha_mask_path, config.resize)

    centers, label_map = design_mosaic(image, density_map, config, debug_dir)

    for aprox_prams in [('poly', 0.05), ('fourier', 0.001), ('square', None)]:
        label_map = simplify_label_map(label_map, aprox_prams, erode_blobs=False)
        render_label_map_with_image(label_map, image, centers,
                                    os.path.join(outputs_dir, f'FinalMosaic_{aprox_prams}.png'))
@dataclass
class MosaicConfig:
    img_path: str = 'images/images/turk.jpg'
    alpha_mask_path: str = 'images/masks/turk_mask.png'
    edges_reference: str = 'image'   # if path: compute edges from image itself or from the mask. else: Canny edge detection
    resize: int = 1024
    n_tiles: int = 3000
    n_iters: int = 10
    edges_mode: str = 'edges' # edges / level_lines

    def get_str(self):
        im_name = os.path.basename(os.path.splitext(self.img_path)[0])
        return f"{im_name}_R-{self.resize}_N-{self.n_tiles}_EM-{self.edges_mode}"


if __name__ == '__main__':
    configs = MosaicConfig()
    make_edge_oriented_vornoi_mosaic(configs, os.path.join("outputs", "Hausner_rectangle_fit", configs.get_str()))

