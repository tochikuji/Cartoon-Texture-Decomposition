from typing import Callable, List, Union, Iterable, Tuple, NewType

import numpy
import cv2

from cartex import expect_valid_float_image
from cartex import iterativeLPF


def imGrad_sobel(img, stride=1):
    dx = cv2.Sobel(img, ddepth=-1, dx=stride, dy=0, borderType=cv2.BORDER_REFLECT)
    dy = cv2.Sobel(img, ddepth=-1, dx=0, dy=stride, borderType=cv2.BORDER_REFLECT)

    norm_im = numpy.sqrt(dx ** 2 + dy ** 2)

    return norm_im


def LTV(img: numpy.ndarray, sigma=2.):
    grad = imGrad_sobel(img)
    ltv_img = cv2.GaussianBlur(img, (5, 5), sigma, borderType=cv2.BORDER_REFLECT)

    return ltv_img


def channelwiseLTV(img: numpy.ndarray, sigma=2.):
    n_channel = img.shape[2]
    ltv_img = numpy.zeros_like(img)

    for k in range(n_channel):
        prid = img[:, :, k]
        ltv_prid = LTV(prid, sigma)

        ltv_img[:, :, k] = ltv_prid

    return ltv_img
