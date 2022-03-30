import cv2
import numpy as np
from scipy import ndimage


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


def create_direction_field(edge_map):
    """
    Create a direction vector field from edges in the edges_map.
    Compute edges map its distance transform and then its gradient map.
    Normalize gradient vector to have unit norm.
    """
    dist_transform = ndimage.distance_transform_edt(edge_map == 0)
    dist_transform = cv2.GaussianBlur(dist_transform, (5, 5), 0)
    direction_field = np.stack(np.gradient(dist_transform), axis=2)
    direction_field = cv2.GaussianBlur(direction_field, (5, 5), 0)
    direction_field = normalize_vector_field(direction_field)

    return direction_field, dist_transform

def normalize_vector_field(vector_field):
    """
    Divide an np array of shape (H,W,d) by its norm in the last axis, replace zero divisions by random vectors
    """
    d = vector_field.shape[-1]
    norms = np.linalg.norm(vector_field, axis=-1)
    vector_field /= norms[..., None]
    if 0 in norms:
        nans = np.where(norms == 0)
        random_direction = np.random.rand(len(nans[0]), d)
        random_direction /= np.linalg.norm(random_direction, axis=1, keepdims=True)
        vector_field[nans] = random_direction

    return vector_field


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