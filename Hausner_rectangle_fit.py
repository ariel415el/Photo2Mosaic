import os.path
from dataclasses import dataclass

from scipy.ndimage import binary_dilation, binary_erosion, binary_closing

from utils import *
from utils import get_edges_map_canny
from utils.tesselation.oriented_tesselation import oriented_tesselation_with_edge_avoidance

ROTATION_MATRICES={x: get_rotation_matrix(x) for x in [45, 90, 135, 225, 315]}


class HausnerMosaicMaker:
    @staticmethod
    def _design_mosaic(img, density_map, config, debug_dir):
        if config.edges_reference and os.path.exists(config.edges_reference):
            edge_map = read_edges_map(config.edges_reference, t=127, resize=img.shape[0])
        else:
            edge_map = get_edges_map_canny(img)

        direction_field, _ = create_direction_field(edge_map)

        centers, label_map = oriented_tesselation_with_edge_avoidance(edge_map, direction_field, density_map, config.n_tiles, config.n_iters, cut_by_edges=True, debug_dir=debug_dir)

        # Debug code:
        plot_vector_field(direction_field, img, path=os.path.join(debug_dir, 'VectorField.png'))
        cv2.imwrite(os.path.join(debug_dir, 'EdgeMap.png'), edge_map * 255)

        return centers, label_map

    @staticmethod
    def _render_tiles(img, centers, label_map, approximation_method='rectangle',path='mosaic.png'):
        """
        Render the mosaic: place oritented squares on a canvas with size defined by the alpha mask
        """
        mosaic = np.ones_like(img) * 127
        coverage_map = np.zeros(img.shape[:2])

        for i, label in enumerate(np.unique(label_map)[1:]):  # First label is -1
            color = img[int(centers[i][0]), int(centers[i][1])]

            blob = (label_map == label)
            # blob = binary_erosion(blob, iterations=1)

            contours, hierarchy = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if approximation_method == "poly":
                    shape = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(contours[0], True), True)
                else: # box
                     shape = np.int0(cv2.boxPoints(cv2.minAreaRect(np.array(cnt).astype(int))))

                cv2.drawContours(mosaic, [shape], -1, color=color.tolist(), thickness=cv2.FILLED)

                # update overidden pixels
                tmp = np.zeros_like(coverage_map)
                cv2.drawContours(tmp, [shape], -1, color=1, thickness=cv2.FILLED)
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

        image, density_map = load_images(config.img_path, config.alpha_mask_path, config.resize)

        centers, label_map = HausnerMosaicMaker._design_mosaic(image, density_map, config, debug_dir)

        # write final mosaic
        HausnerMosaicMaker._render_tiles(image, centers, label_map, approximation_method='poly', path=os.path.join(outputs_dir, f'FinalMosaic_poly.png'))
        HausnerMosaicMaker._render_tiles(image, centers, label_map, approximation_method='rectangle', path=os.path.join(outputs_dir, f'FinalMosaic_rectangle.png'))


@dataclass
class MosaicConfig:
    img_path: str = 'images/images/Yossi2.jpg'
    alpha_mask_path: str = 'images/masks/Yossi2_mask.png'
    edges_reference: str = 'images/edge_maps/Yossi2.png'   # if path: compute edges from image itself or from the mask. else: Canny edge detection
    resize: int = 512
    n_tiles: int = 500
    n_iters: int = 100
    init_mode: str = 'random' # random / uniform

    def get_str(self):
        im_name = os.path.basename(os.path.splitext(self.img_path)[0])
        return f"{im_name}_R-{self.resize}_N-{self.n_tiles}"


if __name__ == '__main__':
    configs = MosaicConfig()
    mosaic_maker = HausnerMosaicMaker.make_mosaic(configs, os.path.join("outputs", "Hausner2001_rectangle_fit", configs.get_str()))

