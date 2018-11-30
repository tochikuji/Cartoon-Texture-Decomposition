from typing import Callable, List, Union, Iterable, Tuple, NewType

import numpy
import cv2

from cartex import expect_valid_float_image


def iterative_decompose(img: numpy.ndarray, sigma: float, n_iter=5, ksize=(5, 5)):
    """
    Decompose the image into high-freq and low-freq parts with iterative [LH]PF algorithm.

    Args:
        img: input image
        sigma: variation of Gaussian
        n_iter: number of iteration
        ksize: kernel size of Gaussian filter

    Returns:
        tuple of decomposed images (low_freq, high_freq)

    """

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
    """
    Iterative LPF

    Args:
        img: input image
        sigma: variation of Gaussian
        n_iter: number of iteration
        ksize: kernel size of Gaussian filter

    Returns:
        result image of LPF
    """
    return iterative_decompose(img, sigma, n_iter, ksize)[0]


def iterativeHPF(img: numpy.ndarray, sigma: float, n_iter=5, ksize=(5, 5)):
    """
    Iterative HPF

    Args:
        img: input image
        sigma: variation of Gaussian
        n_iter: number of iteration
        ksize: kernel size of Gaussian filter

    Returns:
        result image of HPF
    """
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
