import pyreadr
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils.knn import KNN
import matplotlib.pyplot as plt


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
        if people[i] == 0 or people[i] == 1:
            new_labels.append(labels[i])
            new_images.append(images[i])

    new_labels = np.array(new_labels)
    new_images = np.array(new_images)
    return new_images, new_labels


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
    plt.figure(figsize=(8, 8))
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

    # Performing k-means on each digit
    centroids = []
    centroid_labels = []

    for digit in range(10):
        digit_images = []
        for i in range(labels.shape[0]):
            if labels[i] == digit:
                digit_images.append(images[i])

        digit_images = np.array(digit_images)

        kmeans = KMeans(n_clusters=10)
        kmeans.fit(digit_images)
        centroids.append(kmeans.cluster_centers_)
        centroid_labels.append(digit)

    centroids = np.array(centroids)
    centroid_labels = np.array(centroid_labels)

    # Performing k-NN on the centroids
    knn = KNN(k_neighbors=3)
    knn.fit(centroids, centroid_labels)
    pred = knn.predict(centroids)
    print(pred)
    print(centroid_labels)
