import os.path

import cv2
from GPNN_tiling.NN_modules import *
from GPNN_tiling.GPNN import GPNN_mosaic, MosaicReference

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

if __name__ == '__main__':
    with torch.no_grad():
        content_img_path = '/home/ariel/projects/Photo2Mosaic/Photo2Mosaic/images/images/Alexander.jpg'
        content_image = read_image(content_img_path)
        content_image = get_edge_map(content_image)

        mosaic_path = '../images/mosaics/mosaic7_crop.jpg'
        mosaic_reference = MosaicReference(images=[
                                            read_image(mosaic_path, as_edge=True),
                                            read_image(mosaic_path, rotate=1, as_edge=True),
                                            read_image(mosaic_path, rotate=2, as_edge=True),
                                            read_image(mosaic_path, rotate=3, as_edge=True),
                                            ],
                                           tile_size=35
        )

        # NN_module = PytorchNNLowMemory(alpha=0.005, batch_size=256, use_gpu=True, metric='l2')
        NN_module = PytorchNNLowMemory(alpha=None, batch_size=128, use_gpu=True, metric='l2')
        # NN_module = FaissIVFPQ(use_gpu=True)


        GPNN_module = GPNN_mosaic(NN_module,
                                    patch_size=11,
                                    reduced_patch_size=11,
                                    stride=1,
                                    num_steps=5,
                                    pyr_factor=0.75,
                                    coarse_tile_to_patch=1,
                                    output_tile_size_to_dim=0.01)


        GPNN_module.run(content_image, mosaic_reference, debug_dir=f"outputs/{os.path.basename(content_img_path)}_{NN_module}")


