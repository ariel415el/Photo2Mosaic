import os

import cv2
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F


def rescale_img(img, pyr_factor):
    new_w = int(pyr_factor * img.shape[-1])
    new_h = int(pyr_factor * img.shape[-2])
    return transforms.Resize((new_h, new_w), antialias=True)(img)

def blur(img, pyr_factor):
    """Blur image by downscaling and then upscaling it back to original size"""
    if pyr_factor < 1:
        d_img = rescale_img(img, pyr_factor)
        img = transforms.Resize(img.shape[-2:], antialias=True)(d_img)
    return img


def cv2pt(img):
    img = img / 255.
    img = img * 2 - 1
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)
    elif len(img.shape) == 2:
        img = img[None, ]
    img = torch.from_numpy(img).float()

    return img


def get_pyramid(np_img, n_levels, pyr_factor):
    img = cv2pt(np_img)
    assert 0 < pyr_factor < 1

    pyramid = [img]
    for _ in range(n_levels):
        img = rescale_img(img, pyr_factor)
        pyramid = [img] + pyramid

    pyramid = [x.unsqueeze(0) for x in pyramid]
    return pyramid


def extract_patches(src_img, patch_size, stride, reduced_patch_size=None):
    """
    Splits the image to overlapping patches and returns a pytorch tensor of size (N_patches, 3*patch_size**2)
    """
    channels = src_img.shape[1]
    patches = F.unfold(src_img, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0)) # shape (b, c*p*p, N_patches)
    if reduced_patch_size:
        patches = patches.squeeze(dim=0).permute((1, 0)).reshape(-1, channels,  patch_size, patch_size)
        patches = transforms.Resize((reduced_patch_size, reduced_patch_size), antialias=True)(patches)
        patches = patches.reshape(-1, channels * reduced_patch_size ** 2)
    else:
        patches = patches.squeeze(dim=0).permute((1, 0)).reshape(-1, channels * patch_size ** 2)

    return patches


def combine_patches(patches, patch_size, stride, img_shape):
    """
    Combines patches into an image by averaging overlapping pixels
    :param patches: patches to be combined. pytorch tensor of shape (N_patches, 3*patch_size**2)
    :param img_shape: an image of a shape that if split into patches with the given stride and patch_size will give
                      the same number of patches N_patches
    returns an image of shape img_shape
    """
    patches = patches.permute(1,0).unsqueeze(0)
    combined = F.fold(patches, output_size=img_shape[-2:], kernel_size=patch_size, stride=stride)

    # normal fold matrix
    input_ones = torch.ones(img_shape, dtype=patches.dtype, device=patches.device)
    divisor = F.unfold(input_ones, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
    divisor = F.fold(divisor, output_size=img_shape[-2:], kernel_size=patch_size, stride=stride)

    divisor[divisor == 0] = 1.0
    return (combined / divisor).squeeze(dim=0).unsqueeze(0)


def get_gradient_image(pt_img, get_magnitude=True):
    c = pt_img.shape[1]

    pt_img = transforms.GaussianBlur(kernel_size=5, sigma=3)(pt_img)

    sobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    sobel_kernel = torch.tensor(sobel, dtype=torch.float32).expand(1, 1, 3, 3)
    sobel_kernel = sobel_kernel.repeat(1,c,1,1)

    pt_img = torch.nn.functional.pad(pt_img, pad=[1,1,1,1], mode='reflect')

    dy = F.conv2d(pt_img, sobel_kernel, stride=1, groups=1)
    dx = F.conv2d(pt_img, sobel_kernel.transpose(3,2), stride=1, groups=1)

    if get_magnitude:
        return torch.sqrt(dx**2 + dy**2)
    else:
        return torch.atan2(dx, dy)

def aspect_ratio_resize(img, max_dim=256):
    h, w = img.shape[:2]
    if max(h, w) / max_dim > 1:
        img = cv2.blur(img, ksize=(5, 5))

    if w > h:
        h = int(h/w*max_dim)
        w = max_dim
    else:
        w = int(w/h*max_dim)
        h = max_dim

    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(torch.clip(img, -1, 1), path, normalize=True)