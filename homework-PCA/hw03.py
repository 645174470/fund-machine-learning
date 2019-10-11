# -*- coding: utf-8 -*-
"""
File:   hw03.py
Author: Hao Sun
Date:   
Desc:

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

mean = [100,300]
cov = [[1000,500],
      [500,2000]]
num = 100
data0 = np.random.multivariate_normal(mean, cov, num)  # initialize data
plt.title("Original data")
plt.scatter(data0[:, 0], data0[:, 1])  # scatter original data
data = data0 - mean  # Reduce influence of mean
cov_mat = np.cov(data.T)  # compute covariance
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)  # compute eigenvectors and eogenvalues
pca = data.dot(eigen_vecs)  # inner product to transform
# scikit_pca = PCA(n_components = 2, whiten=False)
# y=scikit_pca.fit_transform(data)
covpca = np.cov(pca.T)  # compute covariance
print(covpca)
plt.figure(2)
plt.title("Result of PCA")
plt.scatter(pca[:, 0], pca[:, 1], c='orange')
# sc = StandardScaler()
# whitendata = sc.fit_transform(y)
whitendata = np.zeros((pca.shape[0], pca.shape[1]))
for i in range(pca.shape[1]):
    whitendata[:, i] = pca[:, i]/covpca[i, i]**0.5  # data whitening by divides std of each dimension
plt.figure(3)
plt.title("Result of data whitening")
plt.scatter(whitendata[:, 0], whitendata[:, 1])
covwhitendata = np.cov(whitendata.T)   # compute covariance
print(covwhitendata)
plt.show()