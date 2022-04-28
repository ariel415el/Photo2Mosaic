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
    image, density_map = load_images(config.img_path, config.size_map_path, config.resize)

    os.makedirs(outputs_dir, exist_ok=True)
    debug_dir = os.path.join(outputs_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # centers, label_map = SLIC_superPixels(self.image, self.density_map, configs.n_tiles, configs.n_iters, debug_dir=debug_dir)
    centers, label_map = scikit_SLICO(image, debug_dir)

    for aprox_prams in [('poly', 0.05), ('fourier', 0.001), ('square', None)]:
        label_map = simplify_label_map(label_map, aprox_prams, erode_blobs=False)
        render_label_map_with_image(label_map, image, centers,
                                    os.path.join(outputs_dir, f'FinalMosaic_{aprox_prams}.png'))

@dataclass
class MosaicConfig:
    img_path: str = 'images/images/turk.jpg'
    size_map_path: str = 'images/masks/turk_mask.png'
    resize: int = 1024
    n_tiles: int = 3000
    n_iters: int = 10

    def get_str(self):
        im_name = os.path.basename(os.path.splitext(self.img_path)[0])
        return f"{im_name}_R-{self.resize}_N-{self.n_tiles}"


if __name__ == '__main__':
    configs = MosaicConfig()
    make_slic_mosaics(configs, os.path.join("outputs", "Doyle2019", configs.get_str()))

