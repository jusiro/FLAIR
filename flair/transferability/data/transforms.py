"""
Augmentations for segmentation
"""

from typing import Tuple, Any

import torch
import kornia


class AugmentationsSegmentation(torch.nn.Module):
    def __init__(self):
        super(AugmentationsSegmentation, self).__init__()

        # we define and cache our operators as class members
        self.k1 = kornia.augmentation.ColorJitter(p=0.25, brightness=0.2, contrast=0.2)
        self.k2 = kornia.augmentation.RandomHorizontalFlip(p=0.5)
        self.k3 = kornia.augmentation.RandomAffine(p=0.25, degrees=(-5, 5), scale=(0.9, 1))
        self.k4 = kornia.augmentation.RandomCrop(size=(1024, 1024))

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[Any, Any]:
        img_out = img
        # 1. apply color only in image
        # img_out = self.k1(img_out)
        # 2. apply geometric tranform
        #img_out = self.k4(self.k3(self.k2(img_out)))
        img_out = self.k3(self.k2(img_out))

        # 3. infer geometry params to mask
        # TODO: this will change in future so that no need to infer params
        #mask_out = self.k4(self.k3(self.k2(mask, self.k2._params), self.k3._params), self.k4._params)
        mask_out = self.k3(self.k2(mask, self.k2._params), self.k3._params)

        return img_out, mask_out

