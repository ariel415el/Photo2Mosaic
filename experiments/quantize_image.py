import cv2
import numpy as np

import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import DBSCAN
from skimage.color import label2rgb
from skimage.transform import rescale

def segment_label_map(label_map, img):
    colored_labels = (label2rgb(label_map) * 255).astype(np.uint8)

    quantized_image = np.zeros_like(img)

    for label in np.unique(label_map):
        nwhere = label_map == label
        quantized_image[nwhere] = np.mean(img[nwhere], axis=0)

    quantized_image = quantized_image.astype(np.uint8)

    return quantized_image, colored_labels

def quntize_to_predefined_pallete(img, n_clusters):
    s = np.array([0.99*255] * n_clusters)
    v = np.array([0.99*255] * n_clusters)
    h = np.linspace(0,179, n_clusters)

    pallete = np.stack([h, s, v], axis=1).astype(np.uint8)

    pallete = cv2.cvtColor(pallete[None, ], cv2.COLOR_HSV2RGB)
    pallete = cv2.cvtColor(pallete, cv2.COLOR_RGB2LAB)[0]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            dists = np.linalg.norm(img[r,c] - pallete, axis=1)
            i = np.argmin(dists)
            img[r,c] = pallete[i]

    return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

def quantize_modulu(img, n_clusters=10):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    k = 21
    img = cv2.GaussianBlur(img, (k, k), k)
    b = int(round((255 / n_clusters ** (1/3))))
    img = (img // b)*b
    img += b // 2

    # img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img

def quantize_mean_shift(img, n_clusters, use_space=True):
    return cv2.pyrMeanShiftFiltering(src=img, sp=100, sr=100, maxLevel=3)

def bilateral(img):
    return cv2.bilateralFilter(img, 9,75,75)

def image_histogram_equalization(image, number_bins=256):
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)



def k_means(img, n_clusters=50, use_space=False):
    color_field = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    color_field = cv2.GaussianBlur(color_field, (3, 3), 3)

    if use_space:
        yx_field = np.dstack(np.indices(color_field.shape[:2])) / np.array(color_field.shape[:2])
        feature_field = np.concatenate((color_field, yx_field), axis=-1).reshape([-1, 5])

    else:
        feature_field = color_field.reshape([-1, 3])

    ret, labels, centers = cv2.kmeans(feature_field.astype(np.float32), n_clusters, None,
                                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),
                                      attempts=10, flags=cv2.KMEANS_PP_CENTERS)

    label_map = labels.reshape(img.shape[:2])

    return segment_label_map(label_map, img)

def cluster_DBSCAN(img, use_space=True, n_downscale=2):
    color_field = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    for _ in range(n_downscale):
        color_field = cv2.pyrDown(color_field)

    if use_space:
        yx_field = np.dstack(np.indices(color_field.shape[:2])) / np.array(color_field.shape[:2])
        feature_image = np.concatenate((color_field, yx_field), axis=-1).reshape([-1, 5])
        db = DBSCAN(eps=4, min_samples=4, metric='euclidean', algorithm='auto')

    else:
        db = DBSCAN(eps=0.01, min_samples=50, metric='euclidean', algorithm='auto')
        feature_image = color_field.reshape([-1, 3])

    feature_image / np.max(feature_image, axis=(0, 1))

    rows, cols, chs = color_field.shape

    db.fit(feature_image)
    label_map = db.labels_
    label_map = np.reshape(label_map, [rows, cols])

    # label_map = resize(label_map, img.shape[:2], order=0)
    label_map = cv2.resize((label_map + 1).astype(np.uint8), img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    return segment_label_map(label_map, img)

def chsv2hsv(chsv, range):
    hsv = np.zeros((chsv.shape[0], 3), dtype=np.uint8)
    chsv /= range

    chsv[:, :2] = chsv[:, :2] * 2 - 1
    hue = np.arctan2(chsv[:, 1], chsv[:, 0])
    hue[hue < 0] += 2 * np.pi
    hsv[:, 0] = np.rad2deg(hue) / 2

    hsv[:, 1:] = chsv[:, 2:] * 255

    return hsv

def hsv2chsv(hsv, range=1):
    h_cyclic = np.array([np.cos(np.deg2rad(hsv[:, :, 0] * 2)), np.sin(np.deg2rad(hsv[:, :, 0] * 2))]).transpose(1, 2, 0)
    h_cyclic = (h_cyclic + 1) / 2   # convert from [-1,1] to [0,1]
    sv = hsv[..., 1:] / 255
    hs_cyclic = np.concatenate([h_cyclic, sv], axis=2)

    hs_cyclic *= range

    return hs_cyclic


def MCC(img, n_clusters=50, h_weight=1, s_weight=1, v_weight=1, spatial_weight=1):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    yx_field = np.dstack(np.indices(hsv.shape[:2])) / np.array(hsv.shape[:2])

    chsv_field = hsv2chsv(hsv, range=1)

    features_field = np.concatenate([yx_field, chsv_field], axis=2)

    features_field *= np.array([spatial_weight, spatial_weight, h_weight, h_weight, s_weight, v_weight])[None, :]

    features = features_field.reshape(-1, features_field.shape[-1]).astype(np.float32)

    ret, labels, centers = cv2.kmeans(features, n_clusters, None,
                                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),
                                    attempts=10, flags=cv2.KMEANS_PP_CENTERS)

    label_map = labels.reshape(img.shape[:2])

    return segment_label_map(label_map, img)

def show_images(img, quantized_images):
    n = 1+len(quantized_images)
    plt.figure(figsize=(n*4, 4))
    plt.subplot(1, n, 1)
    plt.imshow(img)
    plt.title("Original_img")
    plt.axis('off')

    for i, (q_img, name) in enumerate(quantized_images):
        n_colors = len(np.unique(q_img.reshape(-1, img.shape[-1]), axis=0))
        plt.subplot(1, n, 2 + i)
        plt.imshow(q_img)
        plt.title(f"Method: {name}, # colors: {n_colors}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # img = cv2.imread('/home/ariel/university/GPDM/images/mosaics/mosaic7.jpg')
    img = cv2.imread('/home/ariel/projects/Photo2Mosaic/Photo2Mosaic/images/images/Alexander.jpg')
    while max(img.shape) > 2000:
        img = cv2.pyrDown(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    quantized_images = []

    n_clusters = 10

    # quantized_img, color_labels = k_means(img, n_clusters=n_clusters)
    # quantized_images.append((quantized_img, 'lab k-means'))
    #
    # quantized_img, color_labels = k_means(img, n_clusters=n_clusters, use_space=True)
    # quantized_images.append((quantized_img, 'lab&space k-means'))
    #
    # quantized_img, color_labels = cluster_DBSCAN(img, n_downscale=1, use_space=False)
    # quantized_images.append((quantized_img, 'DBSCAN'))
    #
    # quantized_img, color_labels = MCC(img, n_clusters=n_clusters, h_weight=1, v_weight=0, s_weight=0, spatial_weight=0)
    # quantized_images.append((quantized_img, 'MCC'))
    #
    # quantized_img = quantize_modulu(img, n_clusters=n_clusters)
    # quantized_images.append((quantized_img, 'modulu'))

    quantized_img = quantize_mean_shift(img, n_clusters=n_clusters)
    quantized_images.append((quantized_img, 'Mean-shift'))

    quantized_img = bilateral(img)
    quantized_images.append((quantized_img, 'Bilateral'))


    # quantized_img = quntize_to_predefined_pallete(img, n_clusters=n_clusters)
    # quantized_images.append((quantized_img, 'pallete'))

    show_images(img, quantized_images)
