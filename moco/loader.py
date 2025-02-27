# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import torch
from torch import Tensor
from torchvision.transforms.transforms import RandomRotation
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class TwoCropsTransformv2:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, angle_range=(-180.0, 180.0)):
        self.base_transform = base_transform
        self.angle_range = angle_range

    def __call__(self, x):
        angle_q = float(torch.empty(1).uniform_(float(self.angle_range[0]), float(self.angle_range[1])).item())  #angle
        angle_k = float(torch.empty(1).uniform_(float(max(angle_q-2.0, -180)), float(min(angle_q+2.0, 180.0))).item())   #angle-1 ~ angle+1
        q = self.base_transform(x, angle_q)
        k = self.base_transform(x, angle_k)
        return [q, k], [angle_q, angle_k]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class RandomRotationv2(RandomRotation):
    def __init__(self, degrees, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None, return_angle=True):
        super(RandomRotationv2, self).__init__(degrees, 
                                                interpolation=interpolation, 
                                                expand=expand, 
                                                center=center, 
                                                fill=fill, 
                                                resample=resample)
        self.return_angle = return_angle

    def forward(self, img, angle):
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        #angle = self.get_params(self.degrees)
        #if self.return_angle:
        #    return F.rotate(img, angle, self.resample, self.expand, self.center, fill), angle
        #else:
            #return F.rotate(img, angle, self.resample, self.expand, self.center, fill)
        return F.rotate(img, angle, self.resample, self.expand, self.center, fill)

class Composev2(Compose):
    def __init__(self, transforms):
        super(Composev2, self).__init__(transforms)

    def __call__(self, img, angle):
        for t in self.transforms:
            if isinstance(t, RandomRotationv2):
                img=t(img, angle)
            else:
                img = t(img)
        return img
