import os

import numpy as np
import torch
from PIL import Image

from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet101
import torchvision.transforms.functional as F
from torchvision.io import read_image

CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS_TO_INDEX = {cls: idx for (idx, cls) in enumerate(CLASSES)}


class segmentor:
    def __init__(self, T=0.5, resize=None):
        self.model = fcn_resnet50(pretrained=True, progress=False)
        # model = deeplabv3_resnet101(pretrained=True)
        self.model.eval()
        self.T = T
        self.resize = resize

    def __call__(self, image_path, T=0.5, resize=1024):
        with torch.no_grad():
            image = read_image(image_path).unsqueeze(0).float()
            image = F.normalize(image, mean=image.mean(axis=(0, 2, 3)), std=image.std(axis=(0, 2, 3)))
            if self.resize:
                image = F.resize(image, resize)
            output = self.model(image)['out']

            output = torch.nn.functional.softmax(output, dim=1)
            most_dominant_class = torch.sum(output[:, 1:], dim=(2,3)).argmax(1).item() + 1  # Discard background
            class_mask = output[0, most_dominant_class]
            class_mask[class_mask > T] = 1
            class_mask[class_mask <= T] = 0

            from scipy.ndimage import binary_closing, binary_dilation
            closed_class_mask = binary_closing(class_mask, iterations=15).astype(np.uint8)

            closed_class_mask[closed_class_mask==1] = 255

            return Image.fromarray(closed_class_mask)


if __name__ == '__main__':
    model = segmentor()
    images_dir = 'data/images'
    outputs_dir = 'data/masks'
    os.makedirs(outputs_dir, exist_ok=True)
    for img_name in os.listdir(images_dir):
        mask = model(os.path.join(images_dir, img_name), T=0.35)
        mask.save(os.path.join(outputs_dir, img_name))