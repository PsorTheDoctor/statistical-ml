import pyreadr
import numpy as np
# from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from utils.pca import PCA


def get_img(flatten_img):
    dim = 18
    img = np.zeros((dim, dim))
    for x in range(dim):
        for y in range(dim):
            img[y, x] = flatten_img[y*dim - (x-1)]
    return img


if __name__ == '__main__':
    data = pyreadr.read_r('data/data_1.Rdata')
    ciphers = np.array(data['ciphers'])
    flatten_images = ciphers[:, 2:]

    # 0-1 normalization
    flatten_images /= 255.

    images = []
    for flatten_img in flatten_images:
        img = get_img(flatten_img)
        images.append(img)

    images = np.asarray(images)
    # plt.imshow(images[0], cmap='gray')
    # plt.show()

    # Gaussian smoothing
    images = gaussian_filter(images, sigma=0.8)
    # plt.imshow(images[0], cmap='gray')
    # plt.show()

    images = images.reshape((2000, 18 * 18))
    pca = PCA(n_components=10)
    pca.fit_transform(images)
