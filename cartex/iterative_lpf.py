from typing import Callable, List, Union, Iterable, Tuple, NewType

import numpy
import cv2

from cartex import expect_valid_float_image


def iterative_decompose(img: numpy.ndarray, sigma: float, n_iter=5, ksize=(5, 5)):
    with expect_valid_float_image(img) as img_:
        im_low = cv2.GaussianBlur(img_, ksize, sigma, borderType=cv2.BORDER_REFLECT)
        im_high = img_ - im_low

        pmin = im_high.min()
        pmax = im_high.max()

        def calib_minmax(im):
            return (im - im.min()) / (im.max() - im.min()) * (pmax - pmin) + pmin

        for _ in range(n_iter):
            im_high = im_high - cv2.GaussianBlur(im_high, ksize, sigma, borderType=cv2.BORDER_REFLECT)
            im_high = calib_minmax(im_high)

        im_low = img_ - im_high
        return im_low, im_high


def iterativeLPF(img: numpy.ndarray, sigma: float, n_iter=5, ksize=(5, 5)):
    return iterative_decompose(img, sigma, n_iter, ksize)[0]


def iterativeHPF(img: numpy.ndarray, sigma: float, n_iter=5, ksize=(5, 5)):
    return iterative_decompose(img, sigma, n_iter, ksize)[1]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = numpy.random.rand(128, 128, 3)
    plt.subplot(331)
    plt.imshow(img)

    for i in range(8):
        plt.subplot(3, 3, i + 2)
        im_low = iterativeLPF(img, 2, i)
        plt.imshow(im_low)

    plt.show()
