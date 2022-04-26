import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage.util import img_as_float
from skimage.filters import gabor_kernel


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='reflect')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i



def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)


def get_filters():
    # prepare filter bank kernels
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels

if __name__ == '__main__':
    kernels = get_filters()

    img = cv2.imread('/home/ariel/Downloads/mosaic7_crop.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

    patch = img[30-5: 30+5, 30-5: 30+5]

    patch_reactions = np.dstack([ndi.convolve(patch, kernel, mode='reflect') for kernel in kernels])
    img_reactions = np.dstack([ndi.convolve(img, kernel, mode='reflect') for kernel in kernels])


    res = np.zeros_like(img)
    for r in range(5, img.shape[0]-5):
        for c in range(5, img.shape[1]-5):
            res[r, c] = np.mean(np.abs(img_reactions[r - 5: r + 5, c - 5: c + 5] - patch_reactions))

    plt.imshow(res)
    plt.show()

