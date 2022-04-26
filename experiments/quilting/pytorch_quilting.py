import math

import cv2
import numpy as np
import torch

from GPNN_tiling.NN_modules import PytorchNNLowMemory
from GPNN_tiling.utils import cv2pt, extract_patches, save_image, rescale_img
import torch.nn.functional as F


class ReferencePatches:
    def __init__(self, pt_img, patch_size):
        self.patches = extract_patches(pt_img, patch_size, stride=1)
        self.NN_module = PytorchNNLowMemory(alpha=None, batch_size=128, use_gpu=True, metric='l2')
        self.NN_module.init_index(self.patches)

        coord_img = np.dstack(np.indices(pt_img.shape[-2:]))
        coord_img = torch.from_numpy(coord_img.transpose(2, 0, 1)).unsqueeze(0).float()
        coord_patches = F.unfold(coord_img, kernel_size=patch_size, dilation=(1, 1), stride=1,
                                 padding=(0, 0))  # shape (b, c*p*p, N_patches)
        coord_patches = coord_patches.squeeze(dim=0).permute((1, 0)).reshape(-1, 2, patch_size ** 2).numpy().transpose(
            0, 2, 1)

        self.patch_locations = coord_patches.mean(1)

    def get_nn(self, patch):
        index = self.NN_module.search(patch)[0]
        return self.patches[index], self.patch_locations[index]


def show_nns(content, texture, cetner):
    content = cv2pt(content)
    texture = cv2pt(texture).unsqueeze(0)
    y, x = cetner
    p = 35
    hp = p//2
    reference = ReferencePatches(texture, p)

    patch = content[..., y - hp:y + hp + 1, x - hp:x + hp + 1]
    flat_patch = patch.reshape(1, 1*p**2)
    ref_patch, ref_location = reference.get_nn(flat_patch)
    ref_patch = ref_patch.reshape(1, p, p)

    save_image(patch, "./patch.png")
    save_image(ref_patch, "./ref.png")

    print(1)

def get_fittest_patch(res, content, reference, patch_size, overlap, y, x):
    h, w = texture.shape[-2:]

    res_patch = res[..., y: y + patch_size, x: x + patch_size]
    content_patch = extract_patches(content, patch_size, stride=1)

    reference


def quilt(content, texture, patch_size):
    h, w, _ = content.shape

    reference = ReferencePatches(texture, patch_size)

    content = cv2pt(content)
    texture = cv2pt(texture)

    overlap = patch_size // 6

    num_patch_h = math.ceil((h - patch_size) / (patch_size - overlap)) + 1 or 1
    num_patch_w = math.ceil((w - patch_size) / (patch_size - overlap)) + 1 or 1

    res = -1 * torch.ones_like(content)

    for i in range(num_patch_h):
        for j in range(num_patch_w):
            y = i * (num_patch_h - overlap)
            x = j * (num_patch_w - overlap)

            patch = get_fittest_patch(res, content, reference, patch_size, overlap, y, x)

            res[y:y + patch_size, x: x + patch_size] = patch


if __name__ == '__main__':
    content = cv2.imread('/home/ariel/Downloads/single_curve.jpg', cv2.IMREAD_GRAYSCALE)
    texture = cv2.imread('/home/ariel/Downloads/mosaic7.jpg', cv2.IMREAD_GRAYSCALE)

    show_nns(content, texture, [145,227])

    # res = quilt(content, texture, 35, 1)
    # io.imshow(res)
    # io.show()
