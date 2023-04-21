from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from utils.load_data import *


X_train, y_train, X_test, y_test = load_all_persons_in_dataset(n_persons=12)

# Ex. 2
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

print('X_train_pca shape:', X_train_pca.shape)
print('y_train shape:', y_train.shape)
print('X_test_pca shape:', X_test_pca.shape)
print('y_test shape:', y_test.shape)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=30)
tree.fit(X_train_pca, y_train)
accuracy = tree.score(X_test_pca, y_test)
print(accuracy)
# plot_tree(tree)
# plt.show()

# Ex. 3
_, labels, images = load_unsplitted_data(n_persons=12)
images_pca = pca.fit_transform(images)

accuracies_wo_pca = []
accuracies_w_pca = []

for _ in range(10):
    # Without PCA
    X_train, y_train, X_test, y_test = cross_validation(images, labels)
    tree.fit(X_train, y_train)
    acc_wo_pca = tree.score(X_test, y_test)
    accuracies_wo_pca.append(acc_wo_pca)

    # With PCA
    X_train_pca, y_train, X_test_pca, y_test = cross_validation(images_pca, labels)
    tree.fit(X_train_pca, y_train)
    acc_w_pca = tree.score(X_test_pca, y_test)
    accuracies_w_pca.append(acc_w_pca)

    print('Acc without PCA:', acc_wo_pca)
    print('Acc with PCA:', acc_w_pca)

print('Mean accuracy without PCA:', round(np.mean(accuracies_wo_pca), 4))
print('Std accuracy without PCA:', round(np.std(accuracies_wo_pca), 4))
print('Mean accuracy with PCA:', round(np.mean(accuracies_w_pca), 4))
print('Std accuracy with PCA:', round(np.std(accuracies_w_pca), 4))
