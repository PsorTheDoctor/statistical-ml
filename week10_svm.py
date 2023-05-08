"""
SVM
all-persons train acc: 0.9404  #0.9382
all-persons test acc: 0.9027  #0.894
disjunct train acc: 0.9363  #0.9387
disjunct test acc: 0.8972  #0.8869

SVM + PCA (0.99 - 146/145 components)  #183
all-persons train acc: 0.9491  #0.9455
all-persons test acc: 0.9121  #0.9002
disjunct train acc: 0.945  #0.9416
disjunct test acc: 0.9046  #0.893
"""
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.decomposition import PCA
from utils.load_data import *


def validate_svm(y_test, y_pred):
    correctly_classified = 0
    count = 0
    for count in range(np.size(y_pred)):
        if y_test[count] == y_pred[count]:
            correctly_classified += 1
        count += 1

    return correctly_classified, count


# def remove_border(img):
#     img[:, 0] = 255
#     img[:, -1] = 255
#     img[0, :] = 255
#     img[-1, :] = 255
#     return img


# All-persons-in dataset
X_train, y_train, X_test, y_test = load_all_persons_in_dataset(
    n_persons=12, split_ratio=0.8, shuffle=True, gaussian_blur=True, verbose=True
)

print('PCA data processing...')
pca = PCA(n_components=0.99, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print('{} components'.format(pca.n_components_))

svm = SVC()
print('SVM fitting...')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_train)
correct, count = validate_svm(y_train, y_pred)
train_acc = round(correct / count, 4)

y_pred = svm.predict(X_test)
print(metrics.classification_report(y_test, y_pred, zero_division=False))
correct, count = validate_svm(y_test, y_pred)
test_acc = round(correct / count, 4)

print('All-persons-in dataset')
print('Train accuracy:', train_acc)
print('Test accuracy:', test_acc)

# Disjunct dataset
X_train, y_train, X_test, y_test = load_disjunct_dataset(
    n_persons=12, split_ratio=0.8, shuffle=True, gaussian_blur=True, verbose=True
)

print('PCA data processing...')
pca = PCA(n_components=0.99)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print('{} components'.format(pca.n_components_))

svm = SVC()
print('SVM fitting...')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_train)
correct, count = validate_svm(y_train, y_pred)
train_acc = round(correct / count, 4)

y_pred = svm.predict(X_test)
print(metrics.classification_report(y_test, y_pred, zero_division=False))
correct, count = validate_svm(y_test, y_pred)
test_acc = round(correct / count, 4)

print('Disjunct dataset')
print('Train accuracy:', train_acc)
print('Test accuracy:', test_acc)
