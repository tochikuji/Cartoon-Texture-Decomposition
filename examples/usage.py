import numpy
import cv2
import matplotlib.pyplot as plt

from cartex import CartoonTextureDecomposition


filename = './assets/lenna_256.jpg'

# Grayscale images
img = cv2.imread(filename, 0)

# create decomposer object
obj = CartoonTextureDecomposition(sigma=2, ksize=7, n_iter=5)

# decompose into cartoon and texture components
cartoon, texture = obj.decompose(img)

# colored ones
img_color = cv2.imread(filename)
cartoon_color, texture_color = obj.decompose(img_color)


mlp = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10), dpi=320)

plt.subplot(321)
plt.title('Original')
plt.axis('off')
plt.imshow(img, cmap='gray')

plt.subplot(323)
plt.title('Cartoon')
plt.axis('off')
plt.imshow(cartoon, cmap='gray')

plt.subplot(325)
plt.title('Texture')
plt.axis('off')
plt.imshow(texture, cmap='gray')

plt.subplot(322)
plt.title('Original')
plt.axis('off')
plt.imshow(mlp(img_color))

plt.subplot(324)
plt.title('Cartoon')
plt.axis('off')
plt.imshow(mlp(cartoon_color))

plt.subplot(326)
plt.title('Texture')
plt.axis('off')
plt.imshow(mlp(texture_color))

plt.tight_layout()
plt.show()
