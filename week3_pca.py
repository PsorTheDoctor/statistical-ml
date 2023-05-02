import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import time

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


def test_pca(images, n_components=20):
    # 0-1 normalization
    images /= 255.

    explained_variances = []
    cumulative_sum = []

    # Covariance matrix
    cov_mat = np.cov(images)

    # Eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_vals = eig_vals[:n_components]
    eig_vecs = eig_vecs[:n_components]
    print('Eigenvalues: {}'.format(eig_vals))
    print('Eigenvectors: {}'.format(eig_vecs))

    # Sorting the vectors according to eigenvalues
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs = eig_pairs[::-1]

    # Percentage of explained variance
    total = sum(eig_vals)
    explained_var_ratio = [i / total for i in sorted(eig_vals, reverse=True)]
    explained_variances.append(explained_var_ratio)
    print('Explained variance ratio: {}'.format(explained_var_ratio))

    cumulative_explained_var = np.cumsum(explained_var_ratio)
    cumulative_sum.append(cumulative_explained_var)
    print('Cumulative explained variance: {}'.format(cumulative_explained_var))

    return explained_variances, cumulative_sum


if __name__ == '__main__':
    # Ex. 2.1.1
    n_components = 15
    images_all_persons, _, _, _ = load_all_persons_in_dataset(n_persons=4)
    explained_variances1, cumulative_sum1 = test_pca(images_all_persons, n_components)

    images_disjunct, _, _, _ = load_disjunct_dataset(n_persons=4)
    explained_variances2, cumulative_sum2 = test_pca(images_disjunct, n_components)

    fig = plt.figure(figsize=(8, 6))
    plt.title('PCA explained variance')
    plt.bar(np.arange(1, n_components + 1, 1) - 0.2, np.ravel(explained_variances1), 0.4,
            label='Explained variance ratios - all-persons-in', color='seagreen')
    plt.bar(np.arange(1, n_components + 1, 1) + 0.2, np.ravel(explained_variances2), 0.4,
            label='Explained variance ratios - disjunct', color='mediumseagreen')
    plt.plot(np.arange(1, n_components + 1, 1), np.ravel(cumulative_sum1),
             label='Cumulative explained variance - all-persons-in', color='red')
    plt.plot(np.arange(1, n_components + 1, 1), np.ravel(cumulative_sum2),
             label='Cumulative explained variance - disjunct', color='blue')
    plt.legend()
    plt.show()
    
    # Ex. 2.1.3
    X_train, y_train, X_test, y_test = load_all_persons_in_dataset(n_persons=1)

    # 0-1 normalization
    X_train /= 255.
    X_test /= 255.

    time_col = []
    for var in [0.8, 0.85, 0.9, 0.95]:
        pca = PCA(n_components=var)
        X_pca = pca.fit_transform(X_train)

        time_row = []
        for k in [3, 4, 5]:
            model = KNN(k_neighbors=3)
            model.fit(X_pca, y_train)
            start = time.time()
            Y_pred = model.predict(X_pca)
            end = time.time()
            t = round(end - start, 2)
            time_row.append(t)
            print('Explained var: {}, time: {}'.format(var, t))

        time_col.append(time_row)

    computation_time = np.array(time_col)
    variances = ['0.8', '0.85', '0.9', '0.95']
    neighbors = ['3 neighbors', '4 neighbors', '5 neighbors']

    fig, ax = plt.subplots()
    im = ax.imshow(computation_time)
    ax.set_xticks(np.arange(3), labels=neighbors)
    ax.set_yticks(np.arange(4), labels=variances)
    plt.title('Computation time')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(4):
        for j in range(3):
            text = ax.text(j, i, computation_time[i, j], ha='center', va='center', color='w')

    fig.tight_layout()
    plt.show()

    # Ex. 2.2
    _, labels, flatten_images = load_unsplitted_data(n_persons=1)

    # Min-Max normalization
    scaler = MinMaxScaler()
    scaler.fit(flatten_images)
    images = scaler.transform(flatten_images)
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
    print('Std accuracy: {}'.format(np.std(minmax_history)))
    print()

    # Z-standarization
    scaler = StandardScaler()
    scaler.fit(flatten_images)
    images = scaler.transform(flatten_images)
    print('Z standarization')

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
    print('Std accuracy: {}'.format(np.std(z_history)))
    print()

    # No normalization
    print('No normalization')

    nonorm_history = []
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
        nonorm_history.append(test_acc)

        print('Test acc: {}'.format(test_acc))

    print('Mean accuracy: {}'.format(np.mean(nonorm_history)))
    print('Std accuracy: {}'.format(np.std(nonorm_history)))
    print()

    fig = plt.figure(figsize=(8, 6))
    plt.xlabel(['Min-Max', 'Z-std', 'None'])
    plt.ylabel('Accuracy')
    plt.bar(np.arange(3), [np.mean(minmax_history), np.mean(z_history), np.mean(nonorm_history)])
    plt.errorbar(np.arange(3), [np.mean(minmax_history), np.mean(z_history), np.mean(z_history)],
                 yerr=[np.std(minmax_history), np.std(z_history), np.std(nonorm_history)],
                 fmt='o', linewidth=2, capsize=50, color='red')
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

    # Plot blurred
    plt.figure(figsize=(9, 3))
    for i, sigma in enumerate(sigma_to_test):
        plt.subplot(1, 3, i + 1)
        img = gaussian_filter(images[1000], sigma=sigma)
        plt.title('Sigma: {}'.format(sigma))
        plt.imshow(np.fliplr(img.reshape(18, 18)), cmap='gray', interpolation='nearest')
    plt.show()
    
    for sigma in sigma_to_test:
        # Gaussian smoothing
        images = gaussian_filter(images, sigma=sigma)
        plt.imshow(images[0], cmap='gray')
        plt.show()
    
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

    smoothing_factors = [0.5, 0.7, 0.9]

    fig = plt.figure(figsize=(8, 6))
    plt.title('Accuracy with different smoothing factors')
    plt.xlabel('Sigma')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(3), labels=smoothing_factors)
    plt.bar(np.arange(3) - 0.2, [98.11, 92.32, 57.55], 0.4,
            color='seagreen', label='Train dataset')
    plt.bar(np.arange(3) + 0.2, [94.3, 83.85, 33.5], 0.4,
            color='mediumseagreen', label='Test dataset')
    plt.errorbar(np.arange(3) - 0.2, [98.11, 92.32, 57.55],
                 yerr=[0.25, 0.24, 0.67],
                 fmt='o', linewidth=2, capsize=25, color='red')
    plt.errorbar(np.arange(3) + 0.2, [94.3, 83.85, 33.5],
                 yerr=[1.25, 1.43, 4.14],
                 fmt='o', linewidth=2, capsize=25, color='red')
    plt.show()
