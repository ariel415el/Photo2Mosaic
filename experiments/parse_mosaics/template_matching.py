import numpy as np


def template_matching(img, template):
    p = template.shape[0]
    # import torch
    # import torch.nn.functional as F
    # with torch.no_grad():
    #     img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
    #     kernel = torch.from_numpy(kernel).permute(2,0,1).unsqueeze(0).float()
    #     reaction_field = F.conv2d(img, kernel)[0].permute(1,2,0).numpy()

    hp = p // 2
    reaction_field = np.zeros(img.shape[:2])
    for r in range(hp, img.shape[0] - hp):
        for c in range(hp, img.shape[1] - hp):
            patch = img[r - hp:r + hp + 1, c - hp:c + hp + 1]
            # reaction_field[r, c] = (patch * template).sum() / np.sqrt(np.power(patch, 2).sum() * np.power(template, 2).sum())
            reaction_field[r, c] = np.abs(patch.mean() - template.mean()).mean()
            # reaction_field[r, c] = np.abs(np.sort(patch.ravel()) - np.sort(template.ravel())).mean()


    return reaction_field