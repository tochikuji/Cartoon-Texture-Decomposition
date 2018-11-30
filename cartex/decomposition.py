from typing import Callable, List, Union, Iterable, Tuple, NewType

import numpy
import cv2

from cartex import expect_valid_float_image
from cartex import iterativeLPF
from cartex import LTV, channelwiseLTV


class CartoonTextureDecomposition:
    """
    Cartoon+Texture decomposition method proposed by Baudes et al.
    """

    def __init__(self, sigma=2, ksize=5, n_iter=5, threshold_fun=None):
        """
        Initialize decomposer with parameters
        Args:
            sigma: variation of the Gaussian filter. sigma=2 by default as with the original paper.
            ksize: kernel size of Gaussian filter. Small ksize make algorithm faster while inprecise.
            n_iter: Iteration of iterative LPF.
            threshold_fun: Specifying soft threshold function [0, 1] -> [0, 1] that must be a BVF.
                           If it is None, partial linear function made with a0, a1 will be adopted.
        """

        self.ksize = (ksize, ksize)
        self.sigma = sigma
        self.n_iter = n_iter
        self.threshold_fun = threshold_fun

        self.mode = 'gray'

    def __mode(self, img: numpy.ndarray):
        """
        Deduce whether the image has colored.
        Args:
            img: image

        """

        if len(img.shape) == 2:
            self.mode = 'gray'
        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                self.mode = 'color'
            elif img.shape[2] == 1:
                raise ValueError(f'image seems to be a grayscaled image, but the shape {img.shape} is not acceptable,'
                                 f'flatten so as to {img.shape[:2]}')
            else:
                raise ValueError(f"only RGB images are supported; but the shape {img.shape} given.")

        else:
            raise ValueError(f"gray-scaled (h, w), or BGR-coloured (h, w, c) must be specified; but {img.shape} given")

    def __rrr_LTV(self, img: numpy.ndarray, lowpassed_img):
        """
        Relative reduction rate of LTV.

        Args:
            img: original image
            lowpassed_img: the image applied a LPF.

        Returns:
            the map of relative reduction rates corresponding to each pixel
        """

        if self.mode == 'gray':
            imLTV = LTV(img, self.sigma)
            imLTV_lpf = LTV(lowpassed_img, self.sigma)
        else:
            imLTV = channelwiseLTV(img, self.sigma)
            imLTV_lpf = channelwiseLTV(lowpassed_img, self.sigma)

        diffLTV = imLTV - imLTV_lpf

        return diffLTV / imLTV

    def soft_threshold(self, a1, a2):
        """
        Get the threshold function.
        If the threshold_fun has specified in __init__, this returns a specified one.
        Otherwise this generate the partial linear BVF with two control point.

        Args:
            a1:
            a2:

        Returns:
            Soft threshold function: [0, 1] -> [0, 1]
        """
        if callable(self.threshold_fun):
            return self.threshold_fun

        else:
            def _soft_threshold(x):
                if x <= a1:
                    return 0
                elif x >= a2:
                    return 1
                else:
                    a = 1 / (a2 - a1)
                    return a * x - a * a1

            return _soft_threshold

    def decompose(self, img: numpy.ndarray, a1=0.25, a2=0.5) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Decompose the image into cartoon part and texture part.

        Args:
            img: Input image which has (h, w) shape grayscaled image or the (h, w, c) shaped colored one.
            a1: first control point of soft threshold
            a2: second control point of soft threshold

        Returns:
            tuple of the image (cartoon, texture)
        """

        with expect_valid_float_image(img) as img_:
            # L_\sigma * f
            im_low = iterativeLPF(img_, sigma=self.sigma, n_iter=self.n_iter, ksize=self.ksize)
            r3_map = self.__rrr_LTV(img, im_low)
            weightmap = numpy.vectorize(self.soft_threshold(a1, a2))(r3_map)

            if self.mode == 'color':
                weightmap = numpy.asarray([weightmap] * 3, dtype=numpy.float32)

            cartoon = weightmap * im_low + (1 - weightmap) * img_
            texture = img_ - cartoon

            return cartoon.astype(numpy.float32), texture.astype(numpy.float32)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    filename = '/home/aiga/Dropbox/misc/testimage/lenna_256.jpg'
    img = cv2.imread(filename, 0)
    obj = CartoonTextureDecomposition(sigma=2, ksize=7)

    cartoon, texture = obj.decompose(img)

    plt.figure(figsize=(10, 10), dpi=320)

    plt.subplot(321)
    plt.imshow(img, cmap='gray')

    plt.subplot(323)
    plt.imshow(cartoon, cmap='gray')

    plt.subplot(325)
    plt.imshow(texture, cmap='gray')

    img_color = cv2.imread(filename)

    cartoon_color, texture_color = obj.decompose(img_color)
    mlp = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

    plt.subplot(322)
    plt.imshow(mlp(img_color))

    plt.subplot(324)
    plt.imshow(mlp(cartoon_color))

    plt.subplot(326)
    plt.imshow(mlp(texture_color))

    plt.show()
