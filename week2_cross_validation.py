import numpy as np
from scipy.stats import mode
from sklearn.utils import shuffle
# from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time

from utils.load_data import *
from utils.knn import KNN
from week1_knn import test_knn, validate


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

    train_no_cross_val = [100.0, 91.4, 92.7, 90.8, 90.0, 89.0, 88.9, 87.1, 86.8, 85.0] #, 84.6, 83.2, 83.2, 82.0, 80.9]
    test_no_cross_val = [87.2, 80.4, 83.5, 81.5, 81.6, 79.7, 80.0, 79.9, 80.0, 79.3] #, 78.6, 77.5, 76.7, 75.9, 76.4]

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
    X_train, y_train, X_test, y_test = load_all_persons_in_dataset(n_persons=4)
    test_knn (X_train, y_train, X_test, y_test,
        neighbors_to_test=10,
        title='All-persons-in dataset'
    )
    # Case 2: Disjunct dataset
    X_train, y_train, X_test, y_test = load_disjunct_dataset(n_persons=4)
    test_knn (X_train, y_train, X_test, y_test,
        neighbors_to_test=10,
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
            model = KNN(k_neighbors=k)
            
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

    computation_time = [
        np.mean([10.67, 9.21, 10.35]),
        np.mean([164.83, 170.91, 155.93]),
        np.mean([1003.39, 1028.85, 1031.13])
    ]

    fig = plt.figure(figsize=(8, 6))
    plt.title('Performance of sample size')
    plt.xlabel('Number of persons')
    plt.ylabel('Computation time')
    plt.bar(['1', '4', '10'], computation_time)
    plt.show()

    # Heat map
    heat_map = np.array([[10.67, 9.21, 10.35],
                         [164.83, 170.91, 155.93],
                         [1003.39, 1028.85, 1031.13]])

    persons = ['1 person', '4 persons', '10 persons']
    neighbors = ['3 neighbors', '4 neighbors', '5 neighbors']

    fig, ax = plt.subplots()
    im = ax.imshow(heat_map)
    ax.set_xticks(np.arange(3), labels=neighbors)
    ax.set_yticks(np.arange(3), labels=persons)
    plt.title('Computation time')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, heat_map[i, j], ha='center', va='center', color='w')

    fig.tight_layout()
    plt.show()
