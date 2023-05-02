import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

from utils.load_data import load_unsplitted_data
from utils.knn import KNN
from week5_kmeans import kmeans_on_each_digit


def load_five_instances_of_each_digit():
    _, labels, images = load_unsplitted_data(n_persons=1)

    new_labels = np.array([])
    new_images = np.array([]).reshape(0, 18 * 18)
    for digit in range(10):
        indices = np.where(labels == digit)[0]
        instances = images[indices[:5]]
        new_labels = np.concatenate((new_labels, np.ones(5) * digit), axis=0)
        new_images = np.concatenate((new_images, instances))

    print(new_images.shape)
    print(new_labels.shape)

    return new_labels, new_images


def validate_knn(y_test, y_pred):
    correctly_classified = 0
    count = 0
    for count in range(np.size(y_pred)):
        if y_test[count] == y_pred[count]:
            correctly_classified += 1
        count += 1

    return correctly_classified, count


if __name__ == '__main__':
    labels, images = load_five_instances_of_each_digit()

    # Hierarchical clustering
    linked = linkage(images, metric='euclidean', method='complete')

    # Ex. 3.2.1
    plt.figure(figsize=(8, 6))
    plt.title('Clustering on a raw dataset')
    plt.yticks([])
    dendrogram(linked, labels=labels, orientation='top', truncate_mode='lastp', p=30,
               leaf_rotation=90.0, leaf_font_size=12.0)
    plt.show()

    cluster = AgglomerativeClustering(n_clusters=10)
    cluster.fit_predict(images)
    print(cluster.labels_)

    # Ex. 3.2.2
    centroids, centroid_labels = kmeans_on_each_digit(images, labels, n_instances_of_each=5)
    print(centroids.shape)
    linked = linkage(centroids, metric='euclidean', method='complete')

    plt.figure(figsize=(8, 6))
    plt.title('Clustering on k-means compressed dataset')
    plt.yticks([])
    dendrogram(linked, labels=centroid_labels, orientation='top', truncate_mode='lastp',
               p=30, leaf_rotation=90.0, leaf_font_size=12.0)
    plt.show()

    cluster.fit_predict(images)
    print(cluster.labels_)

    # Ex. 3.3
    n = len(centroid_labels)
    split_ratio = 0.8

    X_train = centroids[:int(n * split_ratio), :]
    y_train = centroid_labels[:int(n * split_ratio)]
    X_test = centroids[int(n * split_ratio):, :]
    y_test = centroid_labels[int(n * split_ratio):]

    accuracies = []
    for k in range(1, 14):
        knn = KNN(k_neighbors=k)
        knn.fit(centroids, centroid_labels)
        y_pred = knn.predict(X_test)
        correct, count = validate_knn(y_test, y_pred)
        acc = correct / count
        accuracies.append(acc)
        print('Accuracy:', round(acc, 4))

    print('Mean accuracy:', round(np.mean(accuracies), 4))
    print('Std accuracy:', round(np.std(accuracies), 4))
