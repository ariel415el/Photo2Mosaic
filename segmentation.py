import os

import numpy as np
import torch
from PIL import Image

from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet101
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image

CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS_TO_INDEX = {cls: idx for (idx, cls) in enumerate(CLASSES)}


class segmentor:
    def __init__(self, T=0.5, resize=1024):
        self.model = fcn_resnet50(pretrained=True, progress=False)
        # model = deeplabv3_resnet101(pretrained=True)
        self.model.eval()
        self.T = T
        self.resize = resize

    def __call__(self, image_path, T=0.5, resize=1024, class_name='person'):
        with torch.no_grad():
            image = read_image(image_path).unsqueeze(0).float()
            image = F.normalize(image, mean=image.mean(axis=(0, 2, 3)), std=image.std(axis=(0, 2, 3)))
            # image = F.resize(image, resize)
            output = self.model(image)['out']

            output = torch.nn.functional.softmax(output, dim=1)
            person_mask = output[0, CLASS_TO_INDEX[class_name]]
            person_mask[person_mask > T] = 1
            person_mask[person_mask <= T] = 0

            from scipy.ndimage import binary_closing, binary_dilation
            closed_person_mask = binary_closing(person_mask, iterations=15).astype(np.uint8)

            closed_person_mask[closed_person_mask==1] = 255

            path, ext = os.path.splitext(image_path)
            Image.fromarray(closed_person_mask).save(f"{path}_mask.png")


if __name__ == '__main__':
    model = segmentor()
    model("images/images/Leonardo.jpg", resize=1024, T=0.35)
