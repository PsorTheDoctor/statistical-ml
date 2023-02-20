import pyreadr
import numpy as np
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self, k_neighbors):
        self.k_neighbors = k_neighbors

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.n_examples = X_train.shape[0]
        self.n_features = X_train.shape[1]

    def predict(self, X_test):
        self.X_test = X_test
        self.n_test_examples = X_test.shape[0]
        self.n_test_features = X_test.shape[1]

        Y_pred = np.zeros(self.n_test_examples)
        for i in range(self.n_test_examples):
            x = self.X_test[i]
            # Find k nearest neighbors from the current test example.
            # neighbors = np.zeros(self.k)
            neighbors = self.find_neighbors(x)
            # Most frequent class in k neighbors.
            Y_pred[i] = mode(neighbors)[0][0]

        return Y_pred

    def find_neighbors(self, x):
        # Calculate all the euclidean distances between current
        # test example x and training set X_train.
        distances = np.zeros(self.n_examples)
        for i in range(self.n_examples):
            d = self.euclidean(x, self.X_train[i])
            distances[i] = d

        # Sort Y_train according to euclidean distance
        # and store into Y_train_sorted
        indices = distances.argsort()
        Y_train_sorted = self.Y_train[indices]
        return Y_train_sorted[:self.k_neighbors]

    @staticmethod
    def euclidean(x, x_train):
        return np.sqrt(np.sum(np.square(x - x_train)))


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

    labels = ciphers[:, 1:2].flatten()
    images = ciphers[:, 2:]

    print('Labels shape:', labels.shape)
    print('Images shape:', images.shape)

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

    # My implemented model
    model = KNN(k_neighbors=3)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    my_correct, my_count = validate(Y_test, Y_pred)
    my_result = (my_correct / my_count) * 100

    # Built-in sklearn model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    sklearn_correct, sklearn_count = validate(Y_test, Y_pred)
    sklearn_result = (sklearn_correct / sklearn_count) * 100

    print('Accuracy on test set by my model:', my_result)
    print('Accuracy on test set by my sklearn model:', sklearn_result)
