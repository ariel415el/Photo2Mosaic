import os.path

import cv2
from GPNN_tiling.NN_modules import *
from GPNN_tiling.GPNN import GPNN_mosaic, MosaicReference

def read_image(path, rotate=0):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if rotate != 0:
        img = np.rot90(img, rotate)

    return img

if __name__ == '__main__':
    with torch.no_grad():
        content_img_path = '/home/ariel/projects/Photo2Mosaic/Photo2Mosaic/images/images/Alexander.jpg'
        # content_img_path = '/home/ariel/Downloads/white.png'
        content_image = read_image(content_img_path)

        # mosaic_path = '/home/ariel/Downloads/mosaic7_crop.jpg'
        mosaic_path = '/home/ariel/university/GPDM/images/mosaics/mosaic7.jpg'
        # mosaic_path = '/home/ariel/projects/Photo2Mosaic/Photo2Mosaic/images/mosaics/bricks.jpg'
        mosaic_reference = MosaicReference(images=[
                                            read_image(mosaic_path),
                                            read_image(mosaic_path, rotate=1),
                                            # read_image(mosaic_path, rotate=2),
                                            # read_image(mosaic_path, rotate=3),
                                            ],
                                           tile_size=80
        )

        # NN_module = PytorchNNLowMemory(alpha=0.005, batch_size=256, use_gpu=True, metric='l2')
        NN_module = PytorchNNLowMemory(alpha=None, batch_size=128, use_gpu=True, metric='l2')
        # NN_module = FaissIVFPQ(use_gpu=True)


        GPNN_module = GPNN_mosaic(NN_module,
                                    patch_size=9,
                                    stride=1,
                                    num_steps=5,
                                    pyr_factor=0.75,
                                    coarse_tile_to_patch=0.9,
                                    output_tile_size_to_dim=0.02)

        GPNN_module.run(content_image, mosaic_reference, debug_dir=f"outputs/{os.path.basename(content_img_path)}_{NN_module}")


