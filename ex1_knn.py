import pyreadr
import numpy as np
from scipy.stats import mode
from sklearn.utils import shuffle
# from sklearn.neighbors import KNeighborsClassifier
import time
from utils.knn import KNN


# def get_img(img_data):
#     dim = 18
#     img = np.zeros((dim, dim))
#     for x in range(dim):
#         for y in range(dim):
#             img[x, y] = img_data[y*dim - (x-1)]
#
#     return img


def validate(Y_test, Y_pred):
    correctly_classified = 0
    count = 0
    for count in range(np.size(Y_pred)):
        if Y_test[count] == Y_pred[count]:
            correctly_classified += 1
        count += 1

    return correctly_classified, count


if __name__ == '__main__':
    data = pyreadr.read_r('data/data_1.Rdata')
    ciphers = np.array(data['ciphers'])
    print('Dataset shape:', ciphers.shape)

    labels = np.array(ciphers[:, 1:2], dtype=np.uint8).flatten()
    images = ciphers[:, 2:]

    print('Labels shape:', labels.shape)
    print('Images shape:', images.shape)

    # Shuffling before splitting
    images, labels = shuffle(images, labels, random_state=42)

    # Splitting data 50-50 into train and test dataset
    X_train = images[:1000, :]
    Y_train = labels[:1000]
    X_test = images[1000:, :]
    Y_test = labels[1000:]

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)

    # 0-1 normalization
    X_train /= 255.
    X_test /= 255.

    for k in range(1, 10):
        start = time.time()

        # My implemented model
        model = KNN(k_neighbors=k)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        end = time.time()
        t = round(end - start, 2)

        my_correct, my_count = validate(Y_test, Y_pred)
        my_result = (my_correct / my_count) * 100

        # Built-in sklearn model
        # model = KNeighborsClassifier(n_neighbors=k)
        # model.fit(X_train, Y_train)
        # Y_pred = model.predict(X_test)
        # sklearn_correct, sklearn_count = validate(Y_test, Y_pred)
        # sklearn_result = (sklearn_correct / sklearn_count) * 100

        print('Neighbors: {}, accuracy: {}, time: {}'.format(k, my_result, t))
        # print('Accuracy on test set by our model:', my_result)
        # print('Accuracy on test set by sklearn model:', sklearn_result)
