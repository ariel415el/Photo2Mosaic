import cv2
import numpy as np


def set_image_height_without_distortion(img, resize, mode=None):
    return cv2.resize(img, (int(resize * img.shape[1] / img.shape[0]), resize), interpolation=mode)


def image_histogram_equalization(image, number_bins=256):
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def get_rotation_matrix(theta):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    return np.array(((c, -s), (s, c)))


def simplify_contour(cnt, cutoff=0.01):
    """
    Simplifies a contour by trimming its fourier coefficients
    :param: cnt: opencv contour: an numpy array of shape (n,1,2) indicating conour coordinates
    """
    cnt_complex = cnt[:,0,0] + 1j * cnt[:,0,1]

    fourier = np.fft.fftshift(np.fft.fft(cnt_complex))

    spectrum = np.abs(fourier)
    fourier[spectrum < spectrum.max() * cutoff] = 0

    new_cnt_complex = np.fft.ifft(np.fft.ifftshift(fourier))

    new_cnt = np.stack([new_cnt_complex.real, new_cnt_complex.imag], axis=-1).astype(int)[:, None]

    return new_cnt


class StructureTensor:
    """Unweighted structure tensor. Helps understande the typical gradient directions in an neihberhood"""
    def __init__(self, gray_image, t=0.5):
        self.dx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        self.dy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        self.t = t

    def get_structure_tensor(self, mask):
        S = np.zeros((2,2))
        S[0,0] =(self.dx[mask]**2).sum()
        S[1, 1] =(self.dy[mask]**2).sum()
        S[0, 1] = S[1, 0] = (self.dx[mask]*self.dy[mask]).sum()

        w, v = np.linalg.eig(S)
        order = w.argsort()[::-1]
        w = w[order]
        orientation_coherence = np.sqrt((w[0] - w[1]) / (w[0] + w[1] + np.finfo(float).eps))
        if orientation_coherence >= self.t:
            v = v.transpose(1, 0)
            v = v[order]
            return v

        return None
