import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils.load_data import *

_, labels, images = load_unsplitted_data(n_persons=12)

forest = RandomForestClassifier(n_estimators=100)

pca = PCA(n_components=5)
images_pca = pca.fit_transform(images)

accuracies_wo_pca = []
accuracies_w_pca = []

for _ in range(10):
    # Without PCA
    X_train, y_train, X_test, y_test = cross_validation(images, labels)
    forest.fit(X_train, y_train)
    acc_wo_pca = forest.score(X_test, y_test)
    accuracies_wo_pca.append(acc_wo_pca)

    # With PCA
    X_train_pca, y_train, X_test_pca, y_test = cross_validation(images_pca, labels)
    forest.fit(X_train_pca, y_train)
    acc_w_pca = forest.score(X_test_pca, y_test)
    accuracies_w_pca.append(acc_w_pca)

    print('Acc without PCA:', acc_wo_pca)
    print('Acc with PCA:', acc_w_pca)

print('Mean accuracy without PCA:', round(np.mean(accuracies_wo_pca), 4))
print('Std accuracy without PCA:', round(np.std(accuracies_wo_pca), 4))
print('Mean accuracy with PCA:', round(np.mean(accuracies_w_pca), 4))
print('Std accuracy with PCA:', round(np.std(accuracies_w_pca), 4))

# Results
tree_wo_pca = 0.7211
tree_w_pca = 0.251
forest_30estim_wo_pca = 0.9069
forest_30estim_w_pca = 0.3257
forest_100estim_wo_pca = 0.9262
forest_100estim_w_pca = 0.3415

labels=['single tree', 'random forest \n30 estimators', 'random forest \n100 estimators']

fig = plt.figure(figsize=(8, 6))
plt.title('Accuracy')
plt.xticks(np.arange(1, 4), labels=labels)
plt.bar(np.arange(1, 4), [tree_wo_pca, forest_30estim_wo_pca, forest_100estim_wo_pca],
        color='mediumseagreen', label='Raw data')
plt.bar(np.arange(1, 4), [tree_w_pca, forest_30estim_w_pca, forest_100estim_w_pca],
        bottom=np.zeros(3), color='seagreen', label='PCA compressed data')
plt.legend()
plt.show()
