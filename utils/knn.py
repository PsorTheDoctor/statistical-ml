import numpy as np
from scipy.stats import mode


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
