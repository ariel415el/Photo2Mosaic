import cv2
import numpy as np

from utils import simplify_contour


def show_fourier(freq):
    x = np.log(abs(freq) + 0.00000001)
    x -= x.min()
    x /= x.max()
    cv2.imshow("Ad", x)
    cv2.waitKey(0)

def filter_spatial_frequencies(img):
    h,w = img.shape

    fourier = np.fft.fftshift(np.fft.fft2(img))

    show_fourier(fourier)

    filter = np.zeros_like(fourier, dtype=np.uint8)
    r = 30
    filter[int(h/2-r):int(h/2+r), int(w/2-r):int(w/2+r)] = 1
    fourier *= filter

    show_fourier(fourier)

    img = np.fft.ifft2(np.fft.ifftshift(fourier))
    cv2.imshow("Ad",img.real.astype(np.uint8))
    cv2.waitKey(0)


def show_contour(img, cnt):
    new = np.ones_like(img)*255
    new = cv2.drawContours(new, [cnt], -1, color=0, thickness=1)
    cv2.imshow("asd", new)
    cv2.waitKey(0)


def filter_fourier_coordinates(img, cutoff):
    contours, _ = cv2.findContours((img == 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[np.argmax([len(cnt) for cnt in contours])]

    show_contour(img, cnt)

    if cutoff > 0:
        cnt = simplify_contour(cnt, cutoff)

    show_contour(img, cnt)

if __name__ == '__main__':
    img = cv2.imread('/home/ariel/Desktop/shape-smoothing-original-image.png', cv2.IMREAD_GRAYSCALE)
    img[img < 127] = 0
    img[img >= 127] = 255
    filter_spatial_frequencies(img)
    filter_fourier_coordinates(img, 0.001)