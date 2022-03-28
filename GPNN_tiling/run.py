import os.path

import cv2
import math
from experiments.GPNN_tiling.NN_modules import *
from experiments.GPNN_tiling.GPNN import GPNN_mosaic
from experiments.GPNN_tiling.utils import aspect_ratio_resize

def rescale_img(img, factor):
    new_h = int(factor * img.shape[0])
    new_w = int(factor * img.shape[1])
    return cv2.resize(img, (new_w, new_h))

def get_tuple(arr, factor, i):
    return tuple((arr * (factor ** i)).astype(int))

def print_configs():
    input_shape = np.array(input_img.shape[:2])
    mosaic_shape = np.array(mosaic_img.shape[:2])
    pyr_indeices = list(range(pyr_levels + 1))
    f = pyr_factor
    print(f"Making mosaic from with patch size : {patch_size}")
    print(f"\t{os.path.basename(input_img_path)}  {input_img_org.shape[:2]}  ->  {os.path.basename(mosaic_img_path)} {mosaic_img.shape[:2]} p~({mosaic_tile_size}) ")
    print(f"\tContent pyramid: {[get_tuple(input_shape, 1/f, i) for i in pyr_indeices[::-1]]}")
    print(f"\tMosaic pyramid: {[get_tuple(mosaic_shape, f, i) for i in pyr_indeices]}")
    print(f"\tAprox_tile_shapes: {[int(mosaic_tile_size * f ** i) for i in pyr_indeices]}")

    return

if __name__ == '__main__':
    with torch.no_grad():
        input_img_path = '/home/ariel/projects/Photo2Mosaic/Photo2Mosaic/images/images/Alexander.jpg'
        # input_img_path = '/home/ariel/Downloads/elad2.jpeg'
        mosaic_img_path, mosaic_tile_size = '/home/ariel/university/GPDM/images/mosaics/mosaic7.jpg', 35
        # mosaic_img_path, mosaic_tile_size = '/home/ariel/university/GPDM/images/mosaics/mosaic7.jpg', 35
        # mosaic_img_path, mosaic_tile_size = '/home/ariel/university/GPDM/images/mosaics/mosaic3.jpg', 16
        input_img_org = cv2.imread(input_img_path)
        mosaic_img = cv2.imread(mosaic_img_path)

        patch_size = 8
        pyr_factor = 0.75

        coarsest_level_tile_over_patch_size = 0.75  # defines size of tile in reference coarsest pyramid level
        output_tile_size_over_image_dim = 0.01 # defines size of synthesis corasest level size acording to ref tile size

        # Set number of pyramid levels such that tile size in the coarsest level of the mosaic's pyramid will
        # be coarsest_level_tile_over_patch_size*patch_size; t*f^l = c*p ==> l = log,f(c*p/t)
        coarse_lvl_tile_size = coarsest_level_tile_over_patch_size * patch_size
        pyr_levels = round(math.log(coarse_lvl_tile_size / mosaic_tile_size, pyr_factor))

        # Resize Content image so that it will have output_tile_size_over_image_dim relations between its size and
        coarsest_level_tile_size = mosaic_tile_size * (pyr_factor ** pyr_levels)
        synthesis_initial_size = round(coarsest_level_tile_size / output_tile_size_over_image_dim)
        input_img = aspect_ratio_resize(input_img_org, max_dim=synthesis_initial_size)

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        mosaic_img = cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB)

        # NN_module = PytorchNNLowMemory(alpha=0.005, batch_size=256, use_gpu=True, metric='l2')
        # NN_module = PytorchNNLowMemory(alpha=None, batch_size=128, use_gpu=True, metric='l2')
        NN_module = FaissIVFPQ(use_gpu=True)

        print_configs()

        GPNN_module = GPNN_mosaic(NN_module,
                                    patch_size=patch_size,
                                    stride=1,
                                    num_steps=5,
                                    pyr_factor=pyr_factor,
                                    pyr_levels=pyr_levels,
                                    single_iteration_in_first_pyr_level=False)

        GPNN_module.run(input_img, mosaic_img, debug_dir=f"outputs/{os.path.basename(input_img_path)}->{os.path.basename(mosaic_img_path)}_{NN_module}")