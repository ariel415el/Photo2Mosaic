import argparse
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

    if config.edges_mode == 'level_lines':
        h, w = img.shape[:2]
        tile_size = int(np.sqrt(h * w / config.n_tiles))
        edge_map, _, _ = get_level_lines(edge_map, density_map, tile_size)

    centers, label_map = oriented_tesselation_with_edge_avoidance(edge_map, direction_field, density_map,
                                                                  config.n_tiles, config.n_iters,
                                                                  search_area_factor=2, cut_by_edges=True,
                                                                  debug_dir=debug_dir)

    # Debug code:
    plot_vector_field(direction_field, img, path=os.path.join(debug_dir, 'VectorField.png'))
    cv2.imwrite(os.path.join(debug_dir, 'EdgeMap.png'), edge_map * 255)

    return centers, label_map


def make_smoothed_edge_oriented_vornoi_mosaic(config, outputs_dir):
    debug_dir = os.path.join(outputs_dir, "debug")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    image, density_map = load_images(config.img_path, config.density_map_path, config.output_height)

    centers, label_map = design_mosaic(image, density_map, config, debug_dir)

    for aprox_prams in [(None, None), ('poly', 0.05), ('fourier', 0.001), ('square', None), ('contour', None)]:
        new_label_map = simplify_label_map(label_map, aprox_prams, erode_blobs=False)
        render_label_map_with_image(new_label_map, image, centers, os.path.join(outputs_dir, f'FinalMosaic_{aprox_prams}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a mosaic by an oriented vornoi tessealation where cells are then smoothed to have nicer shapes .')
    parser.add_argument('img_path')
    parser.add_argument('--edges_reference', default='auto', help='Infer edge map automaticaly or from a binary mask file (Specify a file path)')
    parser.add_argument('--density_map_path', default=None, help='A binary mask specifying areas to tile with smaller tiles')
    parser.add_argument('--output_height', default=512)
    parser.add_argument('--n_tiles', default=700)
    parser.add_argument('--n_iters', default=20, help='Number of iterations for the tesselation algorithm')
    parser.add_argument('--edges_mode', default='random', help='Avoid only edge map or all level lines in tesselation', choices=['edges', 'level_lines'])

    configs = parser.parse_args()
    im_name = os.path.basename(os.path.splitext(configs.img_path)[0])
    configs.name = f"{im_name}_R-{configs.output_height}_N-{configs.n_tiles}_EM-{configs.edges_mode}"

    make_smoothed_edge_oriented_vornoi_mosaic(configs, os.path.join("outputs", "Hausner_rectangle_fit", configs.name))