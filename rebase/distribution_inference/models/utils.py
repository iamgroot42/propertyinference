import torch as ch
import torch.nn as nn
import cv2 as cv
import numpy as np
import math


# fake relu function
class fakerelu(ch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# Fake-ReLU module wrapper
class FakeReluWrapper(nn.Module):
    def __init__(self, inplace: bool = False):
        super(FakeReluWrapper, self).__init__()
        self.inplace = inplace

    def forward(self, input: ch.Tensor):
        return fakerelu.apply(input)


# identity function
class basic(ch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# identity function module wrapper
class BasicWrapper(nn.Module):
    def __init__(self, inplace: bool = False):
        super(BasicWrapper, self).__init__()
        self.inplace = inplace

    def forward(self, input: ch.Tensor):
        return basic.apply(input)


# CyCNN polar image transform
def polar_transform(images, transform_type='linearpolar'):
    """
    This function takes multiple images, and apply polar coordinate conversion to it.
    """

    (N, C, H, W) = images.shape

    for i in range(images.shape[0]):

        img = images[i].numpy()  # [C,H,W]
        img = np.transpose(img, (1, 2, 0))  # [H,W,C]

        if transform_type == 'logpolar':
            img = cv.logPolar(img, (H // 2, W // 2), W / math.log(W / 2),
                              cv.WARP_FILL_OUTLIERS).reshape(H, W, C)
        elif transform_type == 'linearpolar':
            img = cv.linearPolar(img, (H // 2, W // 2),
                                 W / 2, cv.WARP_FILL_OUTLIERS).reshape(H, W, C)
        img = np.transpose(img, (2, 0, 1))

        images[i] = ch.from_numpy(img)

    return images
