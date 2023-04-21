import numpy as np
# from scipy.stats import mode
# from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time
from utils.load_data import load_all_persons_in_dataset
from utils.knn import KNN


def validate(Y_test, Y_pred):
    correctly_classified = 0
    count = 0
    for count in range(np.size(Y_pred)):
        if Y_test[count] == Y_pred[count]:
            correctly_classified += 1
        count += 1

    return correctly_classified, count


def test_knn(X_train, Y_train, X_test, Y_test, neighbors_to_test=20, title=''):
    # 0-1 normalization
    X_train /= 255.
    X_test /= 255.

    train_acc_history = []
    test_acc_history = []
    train_time_history = []
    test_time_history = []
    k_neighbors = neighbors_to_test

    for k in range(1, k_neighbors + 1):
        # My implemented model
        model = KNN(k_neighbors=k)
        model.fit(X_train, Y_train)

        train_times = []
        test_times = []
        for _ in range(1):
            start = time.time()
            Y_pred_on_train = model.predict(X_train)
            end = time.time()
            train_times.append(round(end - start, 2))

            start = time.time()
            Y_pred_on_test = model.predict(X_test)
            end = time.time()
            test_times.append(round(end - start, 2))

        t_train = np.mean(train_times)
        t_test = np.mean(test_times)

        train_correct, train_count = validate(Y_train, Y_pred_on_train)
        test_correct, test_count = validate(Y_test, Y_pred_on_test)
        train_acc = (train_correct / train_count) * 100
        test_acc = (test_correct / test_count) * 100
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        train_time_history.append(t_train)
        test_time_history.append(t_test)

        # Built-in sklearn model
        # model = KNeighborsClassifier(n_neighbors=k)
        # model.fit(X_train, Y_train)
        # Y_pred = model.predict(X_test)
        # sklearn_correct, sklearn_count = validate(Y_test, Y_pred)
        # sklearn_result = (sklearn_correct / sklearn_count) * 100

        print('Neighbors: {}, train acc: {}, time: {}'.format(k, train_acc, t_train))
        print('Neighbors: {}, test acc: {}, time: {}'.format(k, test_acc, t_test))
        # print('Accuracy on test set by our model:', my_result)
        # print('Accuracy on test set by sklearn model:', sklearn_result)

    fig = plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.bar(np.arange(1, k_neighbors + 1, 1), train_acc_history,
            color='seagreen', label='Train dataset')
    plt.bar(np.arange(1, k_neighbors + 1, 1), test_acc_history,
            bottom=np.zeros(k_neighbors), color='mediumseagreen', label='Test dataset')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel('Number of neighbors')
    plt.ylabel('Time')
    plt.ylim(10, 30)
    plt.plot(np.arange(1, k_neighbors + 1, 1), train_time_history, label='Train dataset')
    plt.plot(np.arange(1, k_neighbors + 1, 1), test_time_history, label='Test dataset')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_all_persons_in_dataset(
         n_persons=1, split_ratio=0.5, shuffle=True, verbose=True
    )
    test_knn(X_train, y_train, X_test, y_test)
