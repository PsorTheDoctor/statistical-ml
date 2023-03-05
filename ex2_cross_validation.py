import pyreadr
import numpy as np
from scipy.stats import mode
from sklearn.utils import shuffle
# from sklearn.neighbors import KNeighborsClassifier
import time
from utils.knn import KNN


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

    # 0-1 normalization
    images /= 255.

    for k in range(1, 10):
        history = []
        print('Neighbors: {}'.format(k))

        # 90-10 cross validation
        test_size = 0.1
        for _ in range(10):
            n = 2000
            images, labels = shuffle(images, labels, random_state=42)
            X_train = images[:int((1 - test_size) * n), :]
            Y_train = labels[:int((1 - test_size) * n)]
            X_test = images[int((1 - test_size) * n):, :]
            Y_test = labels[int((1 - test_size) * n):]

            # My implemented model
            model = KNN(k_neighbors=k)
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)

            my_correct, my_count = validate(Y_test, Y_pred)
            my_result = (my_correct / my_count) * 100

            # Built-in sklearn model
            # model = KNeighborsClassifier(n_neighbors=3)
            # model.fit(X_train, Y_train)
            # Y_pred = model.predict(X_test)
            # sklearn_correct, sklearn_count = validate(Y_test, Y_pred)
            # sklearn_result = (sklearn_correct / sklearn_count) * 100

            print('Accuracy: {}'.format(my_result))
            history.append(my_result)
            # print('Accuracy on test set by our model:', my_result)
            # print('Accuracy on test set by sklearn model:', sklearn_result)

        print('Mean accuracy: {}'.format(np.mean(history)))
        print()
