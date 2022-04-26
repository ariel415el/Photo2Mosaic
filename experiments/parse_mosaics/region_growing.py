import cv2
import numpy as np

from experiments.parse_mosaics.template_matching import template_matching
from scipy import linalg

from experiments.parse_mosaics.utils import show_images


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def region_growing(img, center, t=10):
    p = 7
    region_coords = cur_region_coords = (np.dstack(np.indices((p, p))) + center).reshape(-1, 2).tolist()
    region_values = [img[coord[0], coord[1]] for coord in region_coords]

    while cur_region_coords:
        cur_mean, cur_std = np.mean(region_values, axis=0), np.cov(region_values, rowvar=False)

        neighboring_coords = []
        for coord in cur_region_coords:
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    candidate_coord = [coord[0] + dx, coord[1] + dy]
                    candidate_value = img[candidate_coord[0], candidate_coord[1]]
                    is_new = candidate_coord not in region_coords
                    is_similar = np.linalg.norm(candidate_value - cur_mean) < t
                    if is_new and is_similar :
                        region_coords.append(candidate_coord)
                        neighboring_coords.append(candidate_coord)
        cur_region_coords = neighboring_coords

    seed = cv2.circle(img, (center[1], center[0]), radius=2, thickness=cv2.FILLED, color=[0,0,0])
    mask = np.zeros(img.shape)
    for coord in cur_region_coords:
        mask[coord[0], coord[1]] = [0,255,255]


    show_images([
        (seed, "seed"),
        (0.5 *img + 0.5*mask, "img"),
    ])

if __name__ == '__main__':
    img = cv2.imread('/home/ariel/Downloads/mosaic7_crop.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    region_growing(img, [30,30])