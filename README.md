# Cartoon+Texture Decomposition

## Description

A Python implementation of Cartoon+Texture decomposition algorithm proposed in
 [[Baudes+11]](https://www.ipol.im/pub/art/2011/blmv_ct/article.pdf).
 
## Requirements

- Python > 3.5
- OpenCV
- matplotlib (example)

## Installation

```sh
git clone https://github.com/tochikuji/Cartoon-Texture-Decomposition
cd Cartoon-Texture-Decomposition
pip install .
```

## Example

```python
import cv2
from cartex import CartoonTextureDecomposition 

img = cv2.imread("./lenna.png", 0)
decomposer = CartoonTextureDecomposition()
cartoon, texture = decomposer.decompose(img)
```

Then we get the decomposition result as following (sigma=2):

##### Original
![](https://github.com/tochikuji/GitHub-Assets/raw/master/Cartoon-Texture-Decomposition/lenna_orig.png)
##### Cartoon Part
![](https://github.com/tochikuji/GitHub-Assets/raw/master/Cartoon-Texture-Decomposition/lenna_cartoon.png)
##### Texture Part
![](https://github.com/tochikuji/GitHub-Assets/raw/master/Cartoon-Texture-Decomposition/lenna_texture.png)


## LICENSE

Apache 2.0

Refer to `LICENSE`

## Author

Aiga SUZUKI <tochikuji@gmail.com>