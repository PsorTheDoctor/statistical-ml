import numpy as np
from scipy.stats import mode
from sklearn.utils import shuffle
# from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time

from utils.load_data import *
from utils.knn import KNN
from week1_knn import test_knn


def validate(Y_test, Y_pred):
    correctly_classified = 0
    count = 0
    for count in range(np.size(Y_pred)):
        if Y_test[count] == Y_pred[count]:
            correctly_classified += 1
        count += 1

    return correctly_classified, count


if __name__ == '__main__':
    # Ex. 3
    _, labels, images = load_unsplitted_data(n_persons=1)

    # 0-1 normalization
    images /= 255.

    k_neighbors = 10

    for k in range(1, k_neighbors + 1):
        train_history = []
        test_history = []
        train_std_history = []
        test_std_history = []
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

            model = KNN(k_neighbors=k)

            model.fit(X_train, Y_train)
            Y_pred_on_train = model.predict(X_train)

            model.fit(X_test, Y_train)
            Y_pred_on_test = model.predict(X_test)

            train_correct, train_count = validate(Y_train, Y_pred_on_train)
            test_correct, test_count = validate(Y_test, Y_pred_on_test)
            train_acc = (train_correct / train_count) * 100
            test_acc = (test_correct / test_count) * 100

            train_history.append(train_acc)
            train_history.append(train_history)

            print('Train accuracy: {}'.format(train_acc))
            print('Test accuracy: {}'.format(test_acc))

        train_std = np.std(train_history)
        test_std = np.std(test_history)
        train_std_history.append(train_acc)
        train_std_history.append(train_history)

        print('Mean train accuracy: {}'.format(np.mean(train_history)))
        print('Mean test accuracy: {}'.format(np.mean(test_history)))
        print('Std train accuracy: {}'.format(train_std))
        print('Std test accuracy: {}'.format(test_std))
        print()

    train_no_cross_val = []
    test_no_cross_val = [87.2, 80.4, 83.5, 81.5, 81.6, 79.7, 80.0, 79.9, 80.0, 79.3]

    fig = plt.figure(figsize=(8, 6))
    plt.xlabel('Number of neighbors')
    plt.ylabel('Train accuracy')
    plt.plot(np.arange(1, k_neighbors + 1, 1), train_no_cross_val,
             label='Without cross validation')
    plt.plot(np.arange(1, k_neighbors + 1, 1), train_history,
             label='Cross validation')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    plt.xlabel('Number of neighbors')
    plt.ylabel('Test accuracy')
    plt.plot(np.arange(1, k_neighbors + 1, 1), test_no_cross_val,
             label='Without cross validation')
    plt.plot(np.arange(1, k_neighbors + 1, 1), test_history,
             label='Cross validation')
    plt.legend()
    plt.show()

    # Ex. 4
    # Case 1: All-persons-in dataset
    X_train, y_train, X_test, y_test = load_all_persons_in_dataset(n_persons=10)
    test_knn (X_train, y_train, X_test, y_test,
        neighbors_to_test=15,
        title='All-persons-in dataset'
    )
    # Case 2: Disjunct dataset
    X_train, y_train, X_test, y_test = load_disjunct_dataset(n_persons=10)
    test_knn (X_train, y_train, X_test, y_test,
        neighbors_to_test=15,
        title='Disjunct dataset'
    )

    # Ex. 5
    n_persons_to_test = [1, 4, 10]
    k_neighbours_to_test = [3, 4, 5]

    for p in n_persons_to_test:
        X_train, y_train, X_test, _ = load_all_persons_in_dataset(n_persons=p)

        fit_times = []
        pred_times = []

        for k in k_neighbours_to_test:
            start = time.time()
            model.fit(X_train, y_train)
            end_fit = time.time()
            model.predict(X_test)
            end_pred = time.time()

            fit_time = round(end_fit - start, 2)
            pred_time = round(end_pred - start, 2)
            fit_times.append(fit_time)
            pred_times.append(pred_time)

            print('Persons: {}, neighbors: {}, fit time: {}'.format(p, k, fit_time))
            print('Persons: {}, neighbors: {}, pred time: {}'.format(p, k, pred_time))

    fig = plt.figure(figsize=(8, 6))
    plt.title('Performance of sample size')
    plt.xlabel('Number of persons')
    plt.ylabel('Computation time')
    plt.bar()
    plt.show()
