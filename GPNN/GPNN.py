import math

import numpy as np
from tqdm import tqdm

from GPNN.utils import *


class logger:
    """Keeps track of the levels and steps of optimization. Logs it via TQDM"""
    def __init__(self, n_steps, n_lvls):
        self.n_steps = n_steps
        self.n_lvls = n_lvls
        self.lvl = -1
        self.lvl_step = 0
        self.steps = 0

        total_steps= (self.n_lvls) * self.n_steps
        self.pbar = tqdm(total=total_steps, desc='Starting')

    def step(self):
        self.pbar.update(1)
        self.steps += 1
        self.lvl_step += 1

    def new_lvl(self):
        self.lvl += 1
        self.lvl_step = 0

    def print(self):
        # pass
        self.pbar.set_description(f'Lvl {self.lvl}/{self.n_lvls-1}, step {self.lvl_step}/{self.n_steps}')

    def close(self):
        self.pbar.close()


class MosaicReference:
    def __init__(self, images, tile_size):
        self.tile_size = tile_size
        self.images = images
        self.n_levels = None
        self.pyramids = None
        self.pyr_factor = None

    def get_n_levels(self):
        if self.n_levels is None:
            raise AssertionError("Reference pyramids are not built")
        return self.n_levels

    def get_level_images(self, lvl):
        if self.pyramids is None:
            raise AssertionError("Reference pyramids are not built")
        return [pyr[lvl] for pyr in self.pyramids]

    def build_pyramids(self, n_levels, pyr_factor):
        self.n_levels = n_levels + 1
        self.pyr_factor = pyr_factor
        self.pyramids = [get_pyramid(img, n_levels, pyr_factor) for img in self.images]

def get_tuple(arr, factor, i):
    np_arr = np.array(arr)
    return tuple((np_arr * (factor ** i)).astype(int))


class GPNN_mosaic:
    def __init__(self,
                    NN_module,
                    patch_size: int = 7,
                    reduced_patch_size=None,
                    stride: int = 1,
                    num_steps: int = 10,
                    pyr_factor: float = 0.7,
                    coarse_tile_to_patch=0.75,
                    output_tile_size_to_dim = 0.01):
        """
        @param NN_module: Compares between patches
        @param patch_size: The size of patches to extract
        @param reduced_patch_size: resize patches before comparing them
        @param stride: stride of patch extraction
        @param num_steps: number of replacement steps in each pyramid level
        @param pyr_factor: downscale factor in the pyramid
        @param coarse_tile_to_patch: (Defines how many) desired quatient between the size of tiles in the reference image last pyramid level to patch size
        @param output_tile_size_to_dim: How many real tiles in the final image
        """
        self.NN_module = NN_module
        self.patch_size = patch_size
        self.reduced_patch_size = reduced_patch_size if reduced_patch_size is not None else self.patch_size
        self.stride = stride
        self.num_steps = num_steps
        self.pyr_factor = pyr_factor
        self.coarse_tile_to_patch = coarse_tile_to_patch
        self.output_tile_size_to_dim = output_tile_size_to_dim

    def run(self, content_image, mosaic_reference, debug_dir=None):
        # Build working pyramid
        synthesis_initial_size, pyr_levels = self.compute_num_levels_and_synthesis_size(mosaic_reference.tile_size)

        synthesized_image = self.init_synthesis_image(content_image, synthesis_initial_size)

        mosaic_reference.build_pyramids(pyr_levels, self.pyr_factor)

        self.print_sizes(content_image, synthesized_image, mosaic_reference)
        self.logger = logger(self.num_steps, mosaic_reference.get_n_levels())
        for lvl in range(mosaic_reference.get_n_levels()):
            self.logger.new_lvl()

            if lvl > 0:
                # Upscale last lvl output
                synthesized_image = rescale_img(synthesized_image, 1 / self.pyr_factor)

            value_patches, key_patches = self.collect_patches_from_images(mosaic_reference.get_level_images(lvl))
            lvl_output = self.replace_patches(value_patches, key_patches, queries_image=synthesized_image)

            if debug_dir:
                save_image(synthesized_image, f"{debug_dir}/input_lvl-{lvl}.png")
                save_image(lvl_output, f"{debug_dir}/output_lvl-{lvl}.png")
                for i, img in enumerate(mosaic_reference.get_level_images(lvl)):
                    save_image(img, f"{debug_dir}/reference-{i}_lvl-{lvl}.png")

            synthesized_image = lvl_output

        self.logger.close()
        return synthesized_image

    def compute_num_levels_and_synthesis_size(self,  mosaic_tile_size):
        # Set number of pyramid levels such that tile size in the coarsest level of the mosaic's pyramid will
        # be coarse_tile_to_patch*patch_size; t*f^l = c*p ==> l = log,f(c*p/t)
        coarse_tile_size = self.coarse_tile_to_patch * self.patch_size
        pyr_levels = round(math.log(coarse_tile_size / mosaic_tile_size, self.pyr_factor))

        # Resize Content image so that it will have output_tile_size_over_image_dim relations between its size and
        synthesis_initial_size = round(coarse_tile_size / self.output_tile_size_to_dim)

        return synthesis_initial_size, pyr_levels

    def init_synthesis_image(self, content_image, synthesis_initial_size):

        content_img = aspect_ratio_resize(content_image, max_dim=synthesis_initial_size)
        content_img = cv2pt(content_img).unsqueeze(0)

        return content_img

    def collect_patches_from_images(self, images):
        value_patches = []
        key_patches = []
        for img in images:
            patches = extract_patches(img, self.patch_size, self.stride)
            value_patches.append(patches)
            keys_image = blur(img, self.pyr_factor)
            keys_image = self.apply_pre_compare_transform(keys_image)
            patches = extract_patches(keys_image, self.patch_size, self.stride, self.reduced_patch_size)
            key_patches.append(patches)

        value_patches = torch.cat(value_patches, dim=0)
        key_patches = torch.cat(key_patches, dim=0)

        return value_patches, key_patches

    def apply_pre_compare_transform(self, img):
        # return img
        return torch.mean(img, dim=1, keepdims=True)

    def print_sizes(self, content_image, synth_image, mosaic_reference):
        pyr_indeices = list(range(mosaic_reference.n_levels))[::-1]
        synth_shape = synth_image.shape[-2:]
        print(f"- Resizing input image of shape {content_image.shape[:2]} to {synth_shape}")
        print(f"\tContent pyramid: {[get_tuple(synth_shape, 1/self.pyr_factor, i) for i in pyr_indeices]}")
        for i, pyr in enumerate(mosaic_reference.pyramids):
            print(f"\tMosaic ({i}): {[tuple(pyr[j].shape[-2:]) for j in pyr_indeices]}")
        print(f"\tAprox_tile_shapes: {[int(mosaic_reference.tile_size * self.pyr_factor ** j) for j in pyr_indeices[::-1]]}")

    def replace_patches(self, values, keys, queries_image):
        """
        Repeats n_steps iterations of repalcing the patches in "queries_image" by thier nearest neighbors from "values_image".
        The NN matrix is calculated with "keys" wich are a possibly blurred version of the patches from "values_image"
        :param values: patches to replace the with
        :param keys: patches matching to the value patches used for comparing to queries
        :param queries_image: patches that needs to be replaced
        """
        queries_image_original_shape = queries_image.shape

        self.NN_module.init_index(keys)

        for i in range(self.num_steps):
            queries_image = self.apply_pre_compare_transform(queries_image)

            queries = extract_patches(queries_image, self.patch_size, self.stride, self.reduced_patch_size)
            NNs = self.NN_module.search(queries)

            queries_image = combine_patches(values[NNs], self.patch_size, self.stride, queries_image_original_shape)
            if self.logger:
                self.logger.step()
                self.logger.print()

        return queries_image


# # # Debug NNs
# torchvision.utils.save_image((1 + queries_image) / 2, "./query_image.png")
# torchvision.utils.save_image((keys[NNs][:1000].reshape(-1, 1, self.patch_size, self.patch_size) + 1) / 2, "./keys.png")
# torchvision.utils.save_image((values[NNs][:1000].reshape(-1, 1, self.patch_size, self.patch_size) + 1) / 2,
#                              "./values.png")
# torchvision.utils.save_image((queries[:1000].reshape(-1, 1, self.patch_size, self.patch_size) + 1) / 2, "./queries.png")
# exit()