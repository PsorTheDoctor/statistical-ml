import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        # Covariance matrix
        cov_mat = np.cov(X)

        # Eigenvalues and eigenvectors
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        eig_vals = eig_vals[:self.n_components]
        eig_vecs = eig_vecs[:self.n_components]
        print('Eigenvalues: {}'.format(eig_vals))
        print('Eigenvectors: {}'.format(eig_vecs))

        # Sorting the vectors according to eigenvalues
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        eig_pairs = eig_pairs[::-1]

        # Percentage of explained variance
        total = sum(eig_vals)
        explained_var_ratio = [i / total for i in sorted(eig_vals, reverse=True)]
        print('Explained variance ratio: {}'.format(explained_var_ratio))

        cumulative_explained_var = np.cumsum(explained_var_ratio)
        print('Cumulative explained variance: {}'.format(cumulative_explained_var))
