import pyreadr
import numpy as np
from sklearn import utils


def get_img(flatten_img):
    dim = 18
    img = np.zeros((dim, dim))
    for x in range(dim):
        for y in range(dim):
            img[y, x] = flatten_img[y*dim - (x-1)]
    return img


def load_unsplitted_data(n_persons=10, shuffle=False):
    if n_persons in [1, 4, 10, 12]:
        data = pyreadr.read_r('data/data_{}.Rdata'.format(n_persons))
    else:
        print('n_persons should be 1, 4, 10, or 12.')

    ciphers = np.array(data['ciphers'])
    people = np.array(ciphers[:, 0:1], dtype=np.uint8).flatten()
    labels = np.array(ciphers[:, 1:2], dtype=np.uint8).flatten()
    images = ciphers[:, 2:]
    if shuffle:
        images, labels = utils.shuffle(images, labels, random_state=42)

    return people, labels, images


def load_2d_unsplitted_data(n_persons=10, shuffle=False):
    people, labels, flatten_images = load_unsplitted_data(n_persons, shuffle)
    images = []
    for flatten_img in flatten_images:
        img = get_img(flatten_img)
        images.append(img)

    images = np.asarray(images)
    return people, labels, images


def load_all_persons_in_dataset(n_persons=10, split_ratio=0.8, shuffle=False, verbose=False):
    _, labels, images = load_unsplitted_data(n_persons, shuffle)
    n = images.shape[0]

    X_train = images[:int(n * split_ratio), :]
    y_train = labels[:int(n * split_ratio)]
    X_test = images[int(n * split_ratio):, :]
    y_test = labels[int(n * split_ratio):]

    if verbose:
        print('X_train shape:', X_train.shape)
        print('y_train shape:', y_train.shape)
        print('X_test shape:', X_test.shape)
        print('y_test shape:', y_test.shape)

    return X_train, y_train, X_test, y_test


def load_2d_all_persons_in_dataset(n_persons=10, split_ratio=0.8, shuffle=False, verbose=False):
    _, labels, images = load_2d_unsplitted_data(n_persons, shuffle)
    n = images.shape[0]

    X_train = images[:int(n * split_ratio), :]
    y_train = labels[:int(n * split_ratio)]
    X_test = images[int(n * split_ratio):, :]
    y_test = labels[int(n * split_ratio):]

    if verbose:
        print('X_train shape:', X_train.shape)
        print('y_train shape:', y_train.shape)
        print('X_test shape:', X_test.shape)
        print('y_test shape:', y_test.shape)

    return X_train, y_train, X_test, y_test


def load_disjunct_dataset(n_persons=10, split_ratio=0.8, shuffle=False, verbose=False):
    people, labels, images = load_unsplitted_data(n_persons, shuffle)

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in range(people.shape[0]):
        if people[i] in np.arange(int(split_ratio * n_persons)):
            X_train.append(images[i])
            y_train.append(labels[i])
        else:
            X_test.append(images[i])
            y_test.append(labels[i])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    if verbose:
        print('X_train shape:', X_train.shape)
        print('y_train shape:', y_train.shape)
        print('X_test shape:', X_test.shape)
        print('y_test shape:', y_test.shape)

    return X_train, y_train, X_test, y_test


def load_2d_disjunct_dataset(n_persons=10, split_ratio=0.8, shuffle=False, verbose=False):
    people, labels, images = load_2d_unsplitted_data(n_persons, shuffle)

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in range(people.shape[0]):
        if people[i] in np.arange(int(split_ratio * n_persons)):
            X_train.append(images[i])
            y_train.append(labels[i])
        else:
            X_test.append(images[i])
            y_test.append(labels[i])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    if verbose:
        print('X_train shape:', X_train.shape)
        print('y_train shape:', y_train.shape)
        print('X_test shape:', X_test.shape)
        print('y_test shape:', y_test.shape)

    return X_train, y_train, X_test, y_test


def cross_validation(images, labels, test_size=0.1):
    n = len(labels)
    images, labels = utils.shuffle(images, labels, random_state=42)
    X_train = images[:int((1 - test_size) * n), :]
    y_train = labels[:int((1 - test_size) * n)]
    X_test = images[int((1 - test_size) * n):, :]
    y_test = labels[int((1 - test_size) * n):]
    return X_train, y_train, X_test, y_test
