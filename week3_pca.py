import numpy as np
# from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import time

from utils.pca import PCA
from utils.load_data import *
from utils.knn import KNN


def get_img(flatten_img):
    dim = 18
    img = np.zeros((dim, dim))
    for x in range(dim):
        for y in range(dim):
            img[y, x] = flatten_img[y*dim - (x-1)]
    return img


def validate(Y_test, Y_pred):
    correctly_classified = 0
    count = 0
    for count in range(np.size(Y_pred)):
        if Y_test[count] == Y_pred[count]:
            correctly_classified += 1
        count += 1

    return correctly_classified, count


if __name__ == '__main__':
    # Ex. 2.1.3
    X_train, y_train, X_test, y_test = load_all_persons_in_dataset(n_persons=1)

    # 0-1 normalization
    X_train /= 255.
    X_test /= 255.

    for n in range(10, 20):
        pca = PCA(n_components=n)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.fit_transform(X_test)

        model = KNN(k_neighbors=3)
        model.fit(X_train_pca, y_train)
        start = time.time()
        Y_pred_on_train = model.predict(X_test_pca)
        end = time.time()
        t = round(end - start, 2)

        print('Components: {}, time: {}'.format(n, t))

    # Ex. 2.2
    _, labels, flatten_images = load_unsplitted_data(n_persons=1)

    # Min-Max normalization
    scaler = MinMaxScaler()
    images = scaler.fit(flatten_images)
    print('Min-max normalization')

    minmax_history = []
    test_size = 0.1
    for _ in range(10):
        n = 2000
        images, labels = shuffle(images, labels, random_state=42)
        X_train = images[:int((1 - test_size) * n), :]
        Y_train = labels[:int((1 - test_size) * n)]
        X_test = images[int((1 - test_size) * n):, :]
        Y_test = labels[int((1 - test_size) * n):]

        model = KNN(k_neighbors=3)
        model.fit(X_train, Y_train)
        Y_pred_on_test = model.predict(X_test)
        test_correct, test_count = validate(Y_test, Y_pred_on_test)
        test_acc = (test_correct / test_count) * 100
        minmax_history.append(test_acc)

        print('Test acc: {}'.format(test_acc))

    print('Mean accuracy: {}'.format(np.mean(minmax_history)))
    print()

    # Z-standarization
    scaler = StandardScaler()
    X_norm = scaler.fit(flatten_images)
    print('Z-standarization')

    z_history = []
    test_size = 0.1
    for _ in range(10):
        n = 2000
        images, labels = shuffle(images, labels, random_state=42)
        X_train = images[:int((1 - test_size) * n), :]
        Y_train = labels[:int((1 - test_size) * n)]
        X_test = images[int((1 - test_size) * n):, :]
        Y_test = labels[int((1 - test_size) * n):]

        model = KNN(k_neighbors=3)
        model.fit(X_train, Y_train)
        Y_pred_on_train = model.predict(X_train)
        Y_pred_on_test = model.predict(X_test)
        test_correct, test_count = validate(Y_test, Y_pred_on_test)
        test_acc = (test_correct / test_count) * 100
        z_history.append(test_acc)

        print('Test acc: {}'.format(test_acc))

    print('Mean accuracy: {}'.format(np.mean(z_history)))
    print()

    fig = plt.figure(figsize=(8, 6))
    plt.xlabel(['min-max', 'z-std'])
    plt.ylabel('Accuracy')
    plt.bar(np.mean(minmax_history))
    plt.bar(np.mean(z_history))
    plt.legend()
    plt.show()

    # Ex. 2.3
    _, labels, flatten_images = load_unsplitted_data(n_persons=1)

    # 0-1 normalization
    flatten_images /= 255.

    images = []
    for flatten_img in flatten_images:
        img = get_img(flatten_img)
        images.append(img)

    images = np.asarray(images)
    # plt.imshow(images[0], cmap='gray')
    # plt.show()

    sigma_to_test = [0.5, 0.7, 0.9]

    for sigma in sigma_to_test:
        # Gaussian smoothing
        images = gaussian_filter(images, sigma=sigma)
        # plt.imshow(images[0], cmap='gray')
        # plt.show()
        images = images.reshape((2000, 18 * 18))

        # Cross validation
        train_history = []
        test_history = []
        train_std_history = []
        test_std_history = []
        print('Sigma (smoothing factor): {}'.format(sigma))

        test_size = 0.1
        for _ in range(10):
            n = 2000
            images, labels = shuffle(images, labels, random_state=42)
            X_train = images[:int((1 - test_size) * n), :]
            Y_train = labels[:int((1 - test_size) * n)]
            X_test = images[int((1 - test_size) * n):, :]
            Y_test = labels[int((1 - test_size) * n):]

            model = KNN(k_neighbors=3)
            model.fit(X_train, Y_train)
            Y_pred_on_train = model.predict(X_train)
            Y_pred_on_test = model.predict(X_test)

            train_correct, train_count = validate(Y_train, Y_pred_on_train)
            test_correct, test_count = validate(Y_test, Y_pred_on_test)
            train_acc = (train_correct / train_count) * 100
            test_acc = (test_correct / test_count) * 100

            train_history.append(train_acc)
            test_history.append(test_acc)

            print('Train acc: {}, test acc: {}'.format(train_acc, test_acc))

        train_std = np.std(train_history)
        test_std = np.std(test_history)
        train_std_history.append(train_std)
        train_std_history.append(test_std)

        print('Mean train accuracy: {}'.format(np.mean(train_history)))
        print('Mean test accuracy: {}'.format(np.mean(test_history)))
        print('Std train accuracy: {}'.format(train_std))
        print('Std test accuracy: {}'.format(test_std))
        print()

    fig = plt.figure(figsize=(8, 6))
    plt.title('Accuracy with different smoothing factors')
    plt.xlabel('Sigma')
    plt.ylabel('Accuracy')
    plt.plot(sigma_to_test, train_history, label='Train accuracy')
    plt.plot(sigma_to_test, test_history, label='Test accuracy')
    plt.legend()
    plt.show()
