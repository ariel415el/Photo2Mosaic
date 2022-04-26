import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from skimage.color import rgb2gray
from skimage.filters import gaussian
import heapq
from skimage import io, util
from tqdm import tqdm
from skimage.transform import resize

from GPNN_tiling.utils import extract_patches


def L2OverlapDiff(patch, patchLength, overlap, res, y, x):
    error = 0

    if x > 0:
        left = patch[:, :overlap] - res[y:y + patchLength, x:x + overlap]
        error += np.sum(left ** 2)

    if y > 0:
        up = patch[:overlap, :] - res[y:y + overlap, x:x + patchLength]
        error += np.sum(up ** 2)

    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - res[y:y + overlap, x:x + overlap]
        error -= np.sum(corner ** 2)

    return error


def minCutPath(errors):
    # dijkstra's algorithm vertical
    pq = [(error, [i]) for i, error in enumerate(errors[0])]
    heapq.heapify(pq)

    h, w = errors.shape
    seen = set()

    while pq:
        error, path = heapq.heappop(pq)
        curDepth = len(path)
        curIndex = path[-1]

        if curDepth == h:
            return path

        for delta in -1, 0, 1:
            nextIndex = curIndex + delta

            if 0 <= nextIndex < w:
                if (curDepth, nextIndex) not in seen:
                    cumError = error + errors[curDepth, nextIndex]
                    heapq.heappush(pq, (cumError, path + [nextIndex]))
                    seen.add((curDepth, nextIndex))


def minCutPatch(patch, patchLength, overlap, res, y, x):
    patch = patch.copy()
    dy, dx, _ = patch.shape
    minCut = np.zeros_like(patch, dtype=bool)

    if x > 0:
        left = patch[:, :overlap] - res[y:y+dy, x:x+overlap]
        leftL2 = np.sum(left**2, axis=2)
        for i, j in enumerate(minCutPath(leftL2)):
            minCut[i, :j] = True

    if y > 0:
        up = patch[:overlap, :] - res[y:y+overlap, x:x+dx]
        upL2 = np.sum(up**2, axis=2)
        for j, i in enumerate(minCutPath(upL2.T)):
            minCut[:i, j] = True

    np.copyto(patch, res[y:y+dy, x:x+dx], where=minCut)

    return patch


def bestCorrPatch(texture, corrTexture, patchLength, corrTarget, y, x):
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength))

    corrTargetPatch = corrTarget[y:y + patchLength, x:x + patchLength]
    curPatchHeight, curPatchWidth = corrTargetPatch.shape

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            corrTexturePatch = corrTexture[i:i + curPatchHeight, j:j + curPatchWidth]
            e = corrTexturePatch - corrTargetPatch
            errors[i, j] = np.sum(e ** 2)

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i + curPatchHeight, j:j + curPatchWidth]


def bestCorrOverlapPatch(texture, corrTexture, patchLength, overlap,
                         corrTarget, res, y, x, alpha=0.1, level=0):
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength))

    corrTargetPatch = corrTarget[y:y + patchLength, x:x + patchLength]
    di, dj = corrTargetPatch.shape

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            patch = texture[i:i + di, j:j + dj]
            l2error = L2OverlapDiff(patch, patchLength, overlap, res, y, x)
            overlapError = np.sum(l2error)

            corrTexturePatch = corrTexture[i:i + di, j:j + dj]
            corrError = np.sum((corrTexturePatch - corrTargetPatch) ** 2)

            prevError = 0
            if level > 0:
                prevError = patch[overlap:, overlap:] - res[y + overlap:y + patchLength, x + overlap:x + patchLength]
                prevError = np.sum(prevError ** 2)

            errors[i, j] = alpha * (overlapError + prevError) + (1 - alpha) * corrError

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i + di, j:j + dj]


def transfer(texture, target, patchLength, mode="overlap",
             alpha=0.1, level=0, prior=None, blur=False):
    corrTexture = rgb2gray(texture)
    corrTarget = rgb2gray(target)

    if blur:
        corrTexture = gaussian(corrTexture, sigma=3)
        corrTarget = gaussian(corrTarget, sigma=3)

    # remove alpha channel
    texture = util.img_as_float(texture)[:, :, :3]
    target = util.img_as_float(target)[:, :, :3]

    h, w, _ = target.shape
    overlap = patchLength // 2

    numPatchesHigh = math.ceil((h - patchLength) / (patchLength - overlap)) + 1 or 1
    numPatchesWide = math.ceil((w - patchLength) / (patchLength - overlap)) + 1 or 1

    if level == 0:
        res = np.zeros_like(target)
    else:
        res = prior

    pbar = tqdm(total=numPatchesHigh * numPatchesWide)
    for i in range(numPatchesHigh):
        for j in range(numPatchesWide):
            pbar.set_description(f"lvl-{level}- ({i},{j}) / ({numPatchesHigh},{numPatchesWide})")
            y = i * (patchLength - overlap)
            x = j * (patchLength - overlap)

            if i == 0 and j == 0 or mode == "best":
                patch = bestCorrPatch(texture, corrTexture, patchLength, corrTarget, y, x)
            elif mode == "overlap":
                patch = bestCorrOverlapPatch(texture, corrTexture, patchLength,
                                             overlap, corrTarget, res, y, x)
            elif mode == "cut":
                patch = bestCorrOverlapPatch(texture, corrTexture, patchLength,
                                             overlap, corrTarget, res, y, x,
                                             alpha, level)
                patch = minCutPatch(patch, patchLength, overlap, res, y, x)

            res[y:y + patchLength, x:x + patchLength] = patch

            pbar.update(1)

    return res


def transferIter(texture, target, patchLength, n):
    res = transfer(texture, target, patchLength, mode='cut')
    for i in range(1, n):
        alpha = 0.1 + 0.8 * i / (n - 1)
        patchLength = patchLength * 2 ** i // 3 ** i
        print((alpha, patchLength))
        res = transfer(texture, target, patchLength,
                       alpha=alpha, level=i, prior=res)

    return res


def scale_img(img, scale):
    h, w = img.shape[:2]
    return resize(img, (int(h*scale), int(w*scale)))


if __name__ == '__main__':
    content = cv2.imread('/home/ariel/Downloads/single_curve.jpg')
    texture = cv2.imread('/home/ariel/Downloads/mosaic7_crop.jpg')

    # s = "https://people.eecs.berkeley.edu/~efros/research/quilting/figs/transfer/"
    # content = io.imread(s + "bill-big.jpg")
    # texture = io.imread(s + "rice.gif")

    # content = scale_img(content, 0.5)
    # texture = scale_img(texture, 0.5)

    res = transferIter(texture, content, 35, 1)
    io.imshow(res)
    io.show()