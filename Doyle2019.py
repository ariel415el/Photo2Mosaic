import argparse
import os.path
from dataclasses import dataclass

from utils import *
from utils.tesselation.SLIC import SLIC_superPixels, scikit_SLICO
from utils.tesselation.common import render_label_map_with_image, simplify_label_map


def make_slic_mosaics(config, outputs_dir):
    """
    Use SLIC super pixels to create a mosaic. This is a variation of the algorithm described in:
    'Automated pebble mosaic stylization of images'
    """
    image, density_map = load_images(config.img_path, config.density_map_path, config.output_height)

    os.makedirs(outputs_dir, exist_ok=True)
    debug_dir = os.path.join(outputs_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    centers, label_map = SLIC_superPixels(image, density_map, configs.n_tiles, configs.n_iters, debug_dir=debug_dir)
    # centers, label_map = scikit_SLICO(image, debug_dir)

    for aprox_prams in [('poly', 0.05), ('fourier', 0.001), ('square', None)]:
        label_map = simplify_label_map(label_map, aprox_prams, erode_blobs=False)
        render_label_map_with_image(label_map, image, centers,
                                    os.path.join(outputs_dir, f'FinalMosaic_{aprox_prams}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulating Decorative Mosaics by Doyle 2019.')
    parser.add_argument('img_path')
    parser.add_argument('--edges_reference', default='auto', help='Infer edge map automaticaly or from a binary mask file (Specify a file path)')
    parser.add_argument('--density_map_path', default=None, help='A binary mask specifying areas to tile with smaller tiles')
    parser.add_argument('--output_height', default=512)
    parser.add_argument('--n_tiles', default=700)
    parser.add_argument('--n_iters', default=10, help='Number of iterations for the tesselation algorithm')
    configs = parser.parse_args()

    im_name = os.path.basename(os.path.splitext(configs.img_path)[0])
    configs.name = f"{im_name}_R-{configs.output_height}_N-{configs.n_tiles}"

    make_slic_mosaics(configs, os.path.join("outputs", "Doyle2019", configs.name))
