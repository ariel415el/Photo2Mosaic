import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

def find_kmeans_elbow(vecs, mark_x_value=None):
    rets = []
    n_clusters = [2**i for i in range(2, int(math.log(len(vecs) // 2, 2)))]
    for n in n_clusters:
        ret, labels, centers = cv2.kmeans(vecs, n, None,
                                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),
                                          attempts=10, flags=cv2.KMEANS_PP_CENTERS)

        rets.append(ret)
    plt.plot(n_clusters, rets)
    if mark_x_value is not None:
        plt.axvline(x=mark_x_value, color='r', linestyle='-')

    plt.show()

def extract_patches(img, p ,stride, get_coords):
    pt_img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).float()

    patches = F.unfold(pt_img, kernel_size=p, dilation=(1, 1), stride=stride, padding=(0, 0)) # shape (b, c*p*p, N_patches)
    patches = patches.squeeze(dim=0).permute((1, 0)).reshape(-1, 3 * p * p).numpy()

    return patches

def extract_patch_locations():
    if get_coords:
        coord_img = np.dstack(np.indices(img.shape[:2]))
        pt_coord_img = torch.from_numpy(coord_img.transpose(2,0,1)).unsqueeze(0).float()
        coord_patches = F.unfold(pt_coord_img, kernel_size=p, dilation=(1, 1), stride=stride, padding=(0, 0)) # shape (b, c*p*p, N_patches)
        coord_patches = coord_patches.squeeze(dim=0).permute((1, 0)).reshape(-1, 2, p * p).numpy().transpose(0,2,1)
        patch_locations = coord_patches.mean(1)

        return patches, patch_locations
    else:

def filter_by_clusters(vecs, centers, labels, drop_percentage=0.1):
    filtered_indices = []
    indices = []
    all_indices = np.arange(len(vecs))
    for i in range(len(centers)):
        dists = np.linalg.norm(vecs[labels[:, 0] == i] - centers[i], axis=1)
        indices = np.argsort(dists)
        indices = indices[int(np.ceil(drop_percentage*len(indices))):]
        filtered_indices += all_indices[labels[:, 0] == i][indices].tolist()
        print(len(dists), len(indices))

    print(vecs.shape)
    print(indices.shape)

    return filtered_indices

def show_patch_clusters(locations, labels, bg_img=None, path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    if bg_img is not None:
        alpha=0.5
        plt.imshow(img, alpha=alpha)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.scatter(locations[:, 1], locations[:, 0], c=labels.astype(float), alpha=alpha)

    fig.colorbar(im, cax=cax, orientation='vertical')
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()

def main(img, p, stride, n_clusters, drop_percentage):
    patches, patch_locations = extract_patches(img, p, stride)


    ret, labels, centers = cv2.kmeans(patches, n_clusters, None,
                                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10),
                                      attempts=10, flags=cv2.KMEANS_PP_CENTERS)

    filtered_indices = filter_by_clusters(patches, centers, labels, drop_percentage=drop_percentage)

    filtered_patches = patches[filtered_indices]
    filtered_patch_locations = patch_locations[filtered_indices]

    show_patch_clusters(patch_locations, labels, bg_img=img, path="patch_locations.png")
    show_patch_clusters(filtered_patch_locations, labels[filtered_indices], bg_img=img, path="filtered_patch_locations.png")

    patches = torch.from_numpy(patches.reshape(-1, 3, p, p))
    filtered_patches = torch.from_numpy(filtered_patches.reshape(-1, 3, p, p))
    # torchvision.utils.save_image(patches, "patches.png", normalize=True, nrow=round((img.shape[1] - p + 1) / stride))
    # torchvision.utils.save_image(filtered_patches, "filter_patches.png", normalize=True, nrow=round((img.shape[1] - p + 1) / stride))
    # torchvision.utils.save_image(pt_img, "img.png", normalize=True)


if __name__ == '__main__':
    img = cv2.imread('/home/ariel/Downloads/mosaic7_crop.jpg')
    # img = cv2.imread('/home/ariel/Downloads/File-4_crop.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    p=7
    stride = 2
    n_clusters = 200
    drop_percentage = 0.05

    patches, patch_locations = extract_patches(img, p, stride)
    find_kmeans_elbow(patches, n_clusters)

    main(img, p, stride, n_clusters, drop_percentage)