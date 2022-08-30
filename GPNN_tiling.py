import argparse
import os.path

import cv2
from GPNN.NN_modules import *
from GPNN.GPNN import GPNN_mosaic, MosaicReference


def read_image(path, rotate=0, as_edge=False):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if rotate != 0:
        img = np.rot90(img, rotate)
    if as_edge:
        img = get_edge_map(img)
    return img


def get_edge_map(img):
    img = cv2.GaussianBlur(img, (5,5), 0)
    # edges = cv2.Canny(image=img, threshold1=100, threshold2=200)
    from experiments.parse_mosaics.detect_lines import n_channels_gradient
    magnitudes, angles = n_channels_gradient(img)
    edges = magnitudes.astype(np.uint8)

    return edges

def make_mosaic(configs, outputs_dir):
    use_edges = False
    content_image = read_image(configs.img_path, as_edge=use_edges)

    refernce_images = [
        read_image(configs.mosaic_reference_path, as_edge=use_edges),
    ]
    if configs.augment_reference:
        refernce_images += [
            read_image(configs.mosaic_reference_path, rotate=1, as_edge=use_edges),
            read_image(configs.mosaic_reference_path, rotate=2, as_edge=use_edges),
            read_image(configs.mosaic_reference_path, rotate=3, as_edge=use_edges),
    ]

    mosaic_reference = MosaicReference(refernce_images, tile_size=configs.mosaic_refence_patch_size)

    # NN_module = PytorchNNLowMemory(alpha=0.005, batch_size=256, use_gpu=True, metric='l2')
    NN_module = PytorchNNLowMemory(alpha=None, batch_size=128, use_gpu=True, metric='l2')
    # NN_module = FaissIVFPQ(use_gpu=True)

    GPNN_module = GPNN_mosaic(NN_module,
                              configs.patch_size,
                              configs.reduced_patch_size,
                              configs.stride,
                              configs.num_steps,
                              configs.pyr_factor,
                              configs.coarse_lvl_tile_to_patch,
                              configs.output_tile_size_to_dim)

    result = GPNN_module.run(content_image, mosaic_reference, debug_dir=os.path.join(outputs_dir, "debug"))
    cv2.imwrite(os.path.join(outputs_dir, "FinalMosaic.png"), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use GPNN to create mosaics.')

    # IO
    parser.add_argument('img_path')
    parser.add_argument('mosaic_reference_path', help='A mosaic images from whos patches are to be used')
    parser.add_argument('mosaic_refence_patch_size', type=int, help='Spicify the avg diameter (in pixels) of tiles in the reference mosaic')
    parser.add_argument('--augment_reference', action='store_true', default=False, help='use also Rotated reference mosaic')

    # Design
    parser.add_argument('--coarse_lvl_tile_to_patch', type=int, default=1,
                        help='The quotient between the used patch size and the reference mosaic tile size in the smallest pyramid level'
                             'This implicitly defines the size of the coarsest scale of the pyramid.')
    parser.add_argument('--output_tile_size_to_dim', type=float, default=0.025,
                        help='How many real tiles in the final image This defines the size of the output relatively to '
                             'the reference mosaic tile size')

    # GPNN configs
    parser.add_argument('--patch_size', type=int, default=7, help='GPNN patch size')
    parser.add_argument('--reduced_patch_size', default=None, help='Resize patches before comparing them for checper comparison')
    parser.add_argument('--stride', type=int, default=1, help='GPNN patch extraction stride')
    parser.add_argument('--num_steps', type=int, default=5, help='GPNN steps at each level')
    parser.add_argument('--pyr_factor', type=float, default=0.75, help='GPNN pyramid downscale factor')
    configs = parser.parse_args()


    im_name = os.path.basename(os.path.splitext(configs.img_path)[0])
    configs.name = f"{im_name}_C-{configs.coarse_lvl_tile_to_patch}_TS-{configs.output_tile_size_to_dim}"

    make_mosaic(configs, os.path.join("outputs", "GPNN", configs.name))
