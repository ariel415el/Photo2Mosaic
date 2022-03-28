from tqdm import tqdm

from experiments.GPNN_tiling.utils import *


class logger:
    """Keeps track of the levels and steps of optimization. Logs it via TQDM"""
    def __init__(self, n_steps, n_lvls, single_iteration_in_first_pyr_level):
        self.n_steps = n_steps
        self.n_lvls = n_lvls
        self.lvl = -1
        self.lvl_step = 0
        self.steps = 0
        if single_iteration_in_first_pyr_level:
            total_steps= 1 + (self.n_lvls -1) * self.n_steps
        else:
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

class GPNN_mosaic:
    def __init__(self,
                    NN_module,
                    patch_size: int = 7,
                    stride: int = 1,
                    num_steps: int = 10,
                    pyr_levels: int = 3,
                    pyr_factor: float = 0.7,
                    single_iteration_in_first_pyr_level=True):
        self.NN_module = NN_module
        self.patch_size = patch_size
        self.stride = stride
        self.num_steps = num_steps
        self.pyr_factor = pyr_factor
        self.pyr_levels = pyr_levels
        self.single_iteration_in_first_pyr_level = single_iteration_in_first_pyr_level

    def replace_patches(self, values_image, queries_image, n_steps, keys_blur_factor=1, logger=None):
        """
        Repeats n_steps iterations of repalcing the patches in "queries_image" by thier nearest neighbors from "values_image".
        The NN matrix is calculated with "keys" wich are a possibly blurred version of the patches from "values_image"
        :param values_image: The target patches to extract possible pathces or replacement
        :param queries_image: The synthesized image who's patches are to be replaced
        :param n_steps: number of repeated replacements for each patch
        :param keys_blur_factor: the factor with which to blur the values to get keys (image is downscaled and then upscaled with this factor)
        """
        keys_image = blur(values_image, keys_blur_factor)
        queries_image_original_shape = queries_image.shape

        # keys_image = get_gradient_image(keys_image)
        keys_image = torch.mean(keys_image, dim=1, keepdim=True)

        keys = extract_patches(keys_image, self.patch_size, self.stride)

        self.NN_module.init_index(keys)

        values = extract_patches(values_image, self.patch_size, self.stride)
        for i in range(n_steps):
            # queries_image = get_gradient_image(queries_image)
            queries_image = torch.mean(queries_image, dim=1, keepdim=True)

            queries = extract_patches(queries_image, self.patch_size, self.stride)

            NNs = self.NN_module.search(queries)

            # Debug NNs
            # torchvision.utils.save_image(keys[NNs][:30].reshape(-1, 1, self.patch_size, self.patch_size), "./keys.png",normalize=True)
            # torchvision.utils.save_image(queries[:30].reshape(-1, 1, self.patch_size, self.patch_size), "./queries.png",normalize=True)

            queries_image = combine_patches(values[NNs], self.patch_size, self.stride, queries_image_original_shape)
            if logger:
                logger.step()
                logger.print()

        return queries_image

    def run(self, input_image, mosaic_reference, debug_dir=None):
        synthesized_image = cv2pt(input_image).unsqueeze(0)
        reference_pyramid = get_pyramid(mosaic_reference, self.pyr_levels, self.pyr_factor)

        self.logger = logger(self.num_steps, len(reference_pyramid), self.single_iteration_in_first_pyr_level)

        for lvl, lvl_ref_img in enumerate(reference_pyramid):
            self.logger.new_lvl()

            if lvl > 0:
                # h, w = lvl_ref_img.shape[-2:]
                synthesized_image = rescale_img(synthesized_image, 1/self.pyr_factor)

            lvl_output = self.replace_patches(values_image=lvl_ref_img,
                                                         queries_image=synthesized_image,
                                                         n_steps=1 if (self.single_iteration_in_first_pyr_level and lvl == 0) else self.num_steps,
                                                         keys_blur_factor=1 if (self.single_iteration_in_first_pyr_level and lvl == 0) else self.pyr_factor,
                                                         logger=self.logger)
            if debug_dir:
                save_image(synthesized_image, f"{debug_dir}/input{lvl}.png")
                save_image(lvl_ref_img, f"{debug_dir}/reference{lvl}.png")
                save_image(lvl_output, f"{debug_dir}/output{lvl}.png")

            synthesized_image = lvl_output

        self.logger.close()
        return synthesized_image
