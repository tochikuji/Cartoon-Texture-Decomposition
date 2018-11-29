from typing import Callable, List, Union, Iterable, Tuple, NewType

import numpy
import cv2

from cartex import expect_valid_float_image
from cartex import iterativeLPF
from cartex import LTV, channelwiseLTV


class CartoonTextureDecomposition:
    def __init__(self, sigma=2, ksize=5, n_iter=5, threshold_fun=None):
        self.ksize = (ksize, ksize)
        self.sigma = sigma
        self.n_iter = n_iter
        self.threshold_fun = threshold_fun

        self.mode = 'gray'

    def __mode(self, img: numpy.ndarray):
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
        imLTV = LTV(img, self.sigma)
        imLTV_lpf = LTV(lowpassed_img, self.sigma)

        diffLTV = imLTV - imLTV_lpf

        return diffLTV / imLTV

    def soft_threshold(self, a1, a2):
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
        with expect_valid_float_image(img) as img_:
            # L_\sigma * f
            im_low = iterativeLPF(img_, sigma=self.sigma, n_iter=self.n_iter, ksize=self.ksize)
            r3_map = self.__rrr_LTV(img, im_low)
            weightmap = numpy.vectorize(self.soft_threshold(a1, a2))(r3_map)

            cartoon = weightmap * im_low + (1 - weightmap) * img_
            texture = img_ - cartoon

            return cartoon, texture


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = cv2.imread('/home/aiga/Dropbox/misc/testimage/lenna_256.jpg', 0)
    obj = CartoonTextureDecomposition(sigma=2, ksize=7)

    cartoon, texture = obj.decompose(img)

    plt.figure(figsize=(20, 20), dpi=320)
    plt.subplot(311)
    plt.imshow(img, cmap='gray')

    plt.subplot(312)
    plt.imshow(cartoon, cmap='gray')

    plt.subplot(313)
    plt.imshow(texture, cmap='gray')

    plt.show()
