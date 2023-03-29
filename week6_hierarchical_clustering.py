import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from utils.load_data import load_unsplitted_data


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


if __name__ == '__main__':
    labels, images = load_five_instances_of_each_digit()

    # Hierarchical clustering
    linked = linkage(images, metric='euclidean', method='complete')

    plt.figure(figsize=(8, 6))
    dendrogram(linked, orientation='top', truncate_mode='lastp', p=30,
               leaf_rotation=90.0, leaf_font_size=12.0)
    plt.show()

    cluster = AgglomerativeClustering(n_clusters=10)
    cluster.fit_predict(images)
    print(cluster.labels_)
