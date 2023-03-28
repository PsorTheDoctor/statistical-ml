import pyreadr
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def get_img(flatten_img):
    dim = 18
    img = np.zeros((dim, dim))
    for x in range(dim):
        for y in range(dim):
            img[y, x] = flatten_img[y * dim - (x - 1)]
    return img


def perform_pca(n_components, X, idx=0):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    approx = pca.inverse_transform(X_pca)

    plt.title('{}% variance'.format(n_components))
    plt.imshow(np.fliplr(approx[idx].reshape(18, 18)), cmap='gray', interpolation='nearest')
    plt.xlabel('{} components'.format(pca.n_components_))
    plt.xticks([])


if __name__ == '__main__':
    data = pyreadr.read_r('data/data_12.Rdata')
    ciphers = np.array(data['ciphers'])
    flatten_images = ciphers[:, 2:]
    labels = np.array(ciphers[:, 1:2], dtype=np.uint8).flatten()

    images = []
    for flatten_img in flatten_images:
        img = get_img(flatten_img)
        images.append(img)

    images = np.asarray(images)
    images, labels = shuffle(images, labels, random_state=42)

    # Ex. 2.4.1
    # Plot images
    plt.figure(figsize=(9, 3))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title('Label: {}'.format(labels[i]))
        plt.imshow(np.fliplr(images[i]), cmap='gray')
        plt.xticks([])
    plt.show()

    images = images.reshape((images.shape[0], 18 * 18))

    # Reconstruction
    cov_mat = np.cov(images)
    _, eig_vecs = np.linalg.eig(cov_mat)
    # plt.imshow(eig_vecs[:10], cmap='gray')

    # Plot reconstruction
    plt.figure(figsize=(9, 3))
    for i, explained_var in enumerate([0.8, 0.9, 0.95]):
        plt.subplot(1, 3, i + 1)
        perform_pca(explained_var, images)
    plt.show()
