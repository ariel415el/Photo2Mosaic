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


def get_edges_map_DiBlasi2005(image):
    """
    Compute edges as described in the paper
    """
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_gray = image_histogram_equalization(img_gray)

    # Blur the image for better edge detection
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    mean, std = img_gray.mean(), img_gray.std()
    T = std / 4

    img_gray[np.abs(img_gray - mean) > T] = 1
    img_gray[np.abs(img_gray - mean) <= T] = 0

    edges_map = np.absolute(cv2.Laplacian(img_gray, cv2.CV_64F)).astype(np.uint8)
    # edges_map = binary_opening(edges_map, iterations=1).astype(np.uint8)

    return edges_map


def get_edges_map_canny(image, blur_size=7, sigma=None, t1=50, t2=100):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if sigma is None:
        sigma = int(0.01 * np.mean(image.shape[:2]))

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (blur_size, blur_size), sigma)

    # mask = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    mask = cv2.Canny(image=img_blur, threshold1=t1, threshold2=t2)

    mask[mask==255] = 1

    return mask


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


def simplify_contour(cnt, factor=0.01):
    """
    Simplifies a contour by trimming its fourier coefficients
    :param: cnt: opencv contour: an numpy array of shape (n,1,2) indicating conour coordinates
    """
    cnt_complex = cnt[:,0,0] + 1j * cnt[:,0,1]

    fourier = np.fft.fftshift(np.fft.fft(cnt_complex))

    spectrum = np.abs(fourier)
    fourier[spectrum < spectrum.max() * factor] = 0

    new_cnt_complex = np.fft.ifft(np.fft.ifftshift(fourier))

    new_cnt = np.stack([new_cnt_complex.real, new_cnt_complex.imag], axis=-1).astype(int)[:, None]

    return new_cnt