import pyreadr
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

from utils.knn import KNN
from utils.load_data import cross_validation


def load_2person_dataset():
    data = pyreadr.read_r('data/data_4.Rdata')
    ciphers = np.array(data['ciphers'])
    people = np.array(ciphers[:, 0:1], dtype=np.uint8).flatten()
    labels = np.array(ciphers[:, 1:2], dtype=np.uint8).flatten()
    images = ciphers[:, 2:]

    # print('People IDs:', people)
    # print('People IDs shape:', people.shape)
    # print('Labels shape:', labels.shape)
    # print('Images shape:', images.shape)

    new_labels = []
    new_images = []
    for i in range(people.shape[0]):
        if people[i] == 1 or people[i] == 2:
            new_labels.append(labels[i])
            new_images.append(images[i])

    new_labels = np.array(new_labels)
    new_images = np.array(new_images)
    return new_images, new_labels


def kmeans_on_each_digit(images, labels, n_instances_of_each=5):
    kmeans = KMeans(n_clusters=n_instances_of_each)

    # Performing k-means on each digit
    centroids = []
    centroid_labels = []

    for digit in range(10):
        digit_images = []
        for i in range(labels.shape[0]):
            if labels[i] == digit:
                digit_images.append(images[i])

        digit_images = np.array(digit_images)

        kmeans.fit(digit_images)
        for i in range(len(kmeans.cluster_centers_)):
            centroids.append(kmeans.cluster_centers_[i])
            centroid_labels.append(digit)

    centroids = np.array(centroids)
    centroid_labels = np.array(centroid_labels)
    return centroids, centroid_labels


def validate_knn(y_test, y_pred):
    correctly_classified = 0
    count = 0
    for count in range(np.size(y_pred)):
        if y_test[count] == y_pred[count]:
            correctly_classified += 1
        count += 1

    return correctly_classified, count


if __name__ == '__main__':
    data = pyreadr.read_r('data/data_1.Rdata')
    ciphers = np.array(data['ciphers'])
    labels = np.array(ciphers[:, 1:2], dtype=np.uint8).flatten()
    images = ciphers[:, 2:]

    # 0-1 normalization
    images /= 255.

    # k-means on the whole dataset
    kmeans = KMeans(n_clusters=10)
    # kmeans.fit(flatten_images)
    # pred = kmeans.predict(flatten_images)
    #
    # print('k-means predictons:', pred[:10])
    # print('Ground truth:', labels[:10])

    # Visualisation
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(images)
    kmeans.fit(reduced_data)

    # Calculating the centroids
    centroids = kmeans.cluster_centers_
    label = kmeans.fit_predict(reduced_data)
    unique_labels = np.unique(label)

    # Plotting the clusters
    plt.figure(figsize=(10, 6))
    for i in unique_labels:
        plt.scatter(reduced_data[label == i, 0],
                    reduced_data[label == i, 1],
                    label=i)

    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x',
                s=169, linewidths=3, color='k', zorder=10)
    plt.legend()
    plt.show()
    
    # Ex. 3.1.1
    images, labels = load_2person_dataset()

    print('2-person labels shape:', labels.shape)
    print('2-person images shape:', images.shape)

    centroids, centroid_labels = kmeans_on_each_digit(
        images, labels, n_instances_of_each=100
    )
    k_neighbors = 10
    acc_history_wo_kmeans = []
    acc_history_w_kmeans = []
    time_history_wo_kmeans = []
    time_history_w_kmeans = []

    for k in range(1, k_neighbors + 1):
        knn = KNN(k_neighbors=k)

        accuracies_wo_kmeans = []
        accuracies_w_kmeans = []
        times_wo_kmeans = []
        times_w_kmeans = []

        for _ in range(10):
            X_train, y_train, X_test, y_test = cross_validation(images, labels)

            # Performing k-NN on the raw data
            knn.fit(X_train, y_train)
            start = time.time()
            y_pred = knn.predict(X_test)
            end = time.time()

            correct, count = validate_knn(y_test, y_pred)
            acc_wo_kmeans = (correct / count) * 100
            accuracies_wo_kmeans.append(acc_wo_kmeans)
            times_wo_kmeans.append(end - start)

            X_train, y_train, X_test, y_test = cross_validation(centroids, centroid_labels)

            # Performing k-NN on the centroids
            knn.fit(X_train, y_train)
            start = time.time()
            y_pred = knn.predict(X_test)
            end = time.time()
            correct, count = validate_knn(y_test, y_pred)
            acc_w_kmeans = (correct / count) * 100
            accuracies_w_kmeans.append(acc_w_kmeans)
            times_w_kmeans.append(end - start)

        acc_history_wo_kmeans.append(np.mean(accuracies_wo_kmeans))
        acc_history_w_kmeans.append(np.mean(accuracies_w_kmeans))
        time_history_wo_kmeans.append(np.mean(times_wo_kmeans))
        time_history_w_kmeans.append(np.mean(times_w_kmeans))

        print('Neighbors: {}'.format(k))
        print('Mean accuracy without k-means:', round(np.mean(accuracies_wo_kmeans), 4))
        print('Std accuracy without k-means:', round(np.std(accuracies_wo_kmeans), 4))
        print('Mean time without k-means:', round(np.mean(times_wo_kmeans), 4))
        print('Mean accuracy with k-means:', round(np.mean(accuracies_w_kmeans), 4))
        print('Std accuracy with k-means:', round(np.std(accuracies_w_kmeans), 4))
        print('Mean time with k-means:', round(np.mean(times_w_kmeans), 4))
        print()

    fig = plt.figure(figsize=(8, 6))
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.bar(np.arange(1, k_neighbors + 1, 1), acc_history_wo_kmeans,
            color='mediumseagreen', label='Raw data')
    plt.bar(np.arange(1, k_neighbors + 1, 1), acc_history_w_kmeans,
            bottom=np.zeros(k_neighbors), color='seagreen', label='k-means compressed data')
    plt.legend()
    plt.show()

    # time_history_wo_kmeans = [16.9651, 17.0801, 17.2346, 16.9508, 16.7502, 16.7154, 16.4827, 16.9897, 16.6858, 16.7309]
    # time_history_w_kmeans = [1.0846, 1.0552, 1.1189, 1.0411, 1.0459, 1.1232, 1.0533, 1.0505, 1.0505, 1.0364]

    fig = plt.figure(figsize=(8, 6))
    plt.xlabel('Number of neighbors')
    plt.ylabel('Time')
    # plt.ylim(0, 15)
    plt.plot(np.arange(1, k_neighbors + 1, 1), time_history_wo_kmeans,
             label='Raw data')
    plt.plot(np.arange(1, k_neighbors + 1, 1), time_history_w_kmeans,
             label='k-means compressed data')
    plt.legend()
    plt.show()
