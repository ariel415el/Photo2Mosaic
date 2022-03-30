import os.path
from dataclasses import dataclass

import torch

from utils import *
from utils.tesselation import SLIC


class SLICMosaicMaker:
    """
    Use SLIC super pixels to create a mosaic. This is a variation of the algorithm described in:
    'Automated pebble mosaic stylization of images'
    """
    def __init__(self, configs):
        self.config = configs
        self.image, self.density_map = load_images(self.config.img_path, self.config.size_map_path, self.config.resize)

    def make(self, outputs_dir):
        os.makedirs(outputs_dir, exist_ok=True)
        debug_dir = os.path.join(outputs_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        tesselator = SLIC(self.config.n_tiles, self.config.m, self.config.init_mode)
        centers, label_map = tesselator.tesselate(self.image, density_map=self.density_map, n_iters=self.config.n_iters, debug_dir=debug_dir)

        # write final mosaic
        self._render_tiles(centers, label_map, path=os.path.join(outputs_dir, f'FinalMosaic.png'))

    def _render_tiles(self, centers, vornoi_diagram,  path='mosaic.png'):
        """
        Render the mosaic: place oritented squares on a canvas with size defined by the alpha mask
        """
        mosaic = np.ones_like(self.image) * 127
        coverage_map = np.zeros(self.image.shape[:2])
        for i in range(len(centers)):
            mask = (vornoi_diagram == i).astype(np.uint8)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if self.config.contour_approx_method == 'poly':
                    epsilon = self.config.contour_approx_param * cv2.arcLength(cnt, True)
                    cnt = cv2.approxPolyDP(cnt, epsilon, True)
                elif self.config.contour_approx_method == 'fourier':
                    cnt = simplify_contour(cnt, self.config.contour_approx_param)

                color = self.image[int(centers[i][0]), int(centers[i][1])]

                mosaic = cv2.drawContours(mosaic, [cnt], -1, color=color.tolist(), thickness=cv2.FILLED)

                # update overidden pixels
                tmp = np.zeros_like(coverage_map)
                cv2.drawContours(tmp, [cnt], -1, color=1, thickness=cv2.FILLED)
                coverage_map += tmp

        coverage_percentage = (coverage_map > 0).sum() / coverage_map.size * 100
        overlap_percentage = (coverage_map > 1).sum() / coverage_map.size * 100
        print(f"Coverage: {coverage_percentage:.3f}% Overlap:{overlap_percentage:.3f}%)")
        cv2.imwrite(path, mosaic)

        return coverage_percentage, overlap_percentage

@dataclass
class MosaicConfig:
    img_path: str = 'images/images/Alexander_body.jpg'
    size_map_path: str = None
    resize: int = 512
    n_tiles: int = 500
    n_iters: int = 10
    init_mode: str = "uniform"              # random / uniform
    m: int = 1                              # smaller m means stricter color adherence
    contour_approx_method: str = "fourier"  # 'fourier'/'poly'
    contour_approx_param: float = 0.001     # coefficient cutoff percent / allowed error percent

    def get_str(self):
        im_name = os.path.basename(os.path.splitext(self.img_path)[0])
        return f"{im_name}_R-{self.resize}_N-{self.n_tiles}"


if __name__ == '__main__':
    device: torch.device = torch.device("cpu")
    configs = MosaicConfig()
    mosaic_maker = SLICMosaicMaker(configs)
    mosaic_maker.make(os.path.join("outputs", "Doyle2019", configs.get_str()))

