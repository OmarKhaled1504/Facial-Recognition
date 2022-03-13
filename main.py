import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

D = []
y = []
max_accuracy = []


def cal_accuracy(y_test, y_pred):
    print("Accuracy: ",
          accuracy_score(y_test, y_pred) * 100)


def knn(x_train, x_test, y_train, y_test, a, b):
    scores_list = []
    k_range = [1, 3, 5, 7]
    if (a != 0) & (b != 0):
        print("alpha=", a, "\nnumber of features = ", b)
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train.ravel())
        y_pred = knn.predict(x_test)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))
        print("Accuracy Report for k=", k)
        cal_accuracy(y_test, y_pred)
    print("Most accurate k: {}, with Accuracy {}".format(scores_list.index(max(scores_list)) + 1,
                                                         scores_list[scores_list.index(max(scores_list))] * 100))
    max_accuracy.append(max(scores_list) * 100)

    plt.plot(k_range, scores_list)
    plt.title('K-NN with K tuned')
    plt.xlabel("Value of K")
    plt.ylabel("Testing Accuracy")
    plt.show()


def lda(d_train, y_train, d_test, y_test):
    split_ratio = int((d_train.shape[0] / (d_train.shape[0] + d_test.shape[0])) * 10)
    overall_mean = np.mean(d_train, axis=0).reshape(-1, 1)
    sample_mean = np.zeros((40, d_train.shape[1]), dtype=np.float32)
    data = np.vsplit(d_train, d_train.shape[0] / split_ratio)
    for i in range(0, int(d_train.shape[0] / split_ratio)):
        sample_mean[i] = np.mean(data[i], axis=0)
    # print(overall_mean)
    Sb = np.zeros((d_train.shape[1], d_train.shape[1]), dtype=np.float32)
    # print(sample_mean.shape)
    for i in range(0, 40):
        Sb += (split_ratio * np.dot((sample_mean[i].reshape(-1, 1) - overall_mean),
                                    (sample_mean[i].reshape(-1, 1) - overall_mean).transpose()))

    z = np.zeros((40, split_ratio, d_train.shape[1]), dtype=np.float32)
    for i in range(0, 40):
        z[i] = np.subtract(data[i], sample_mean[i].reshape(1, -1))

    s = np.zeros((d_train.shape[1], d_train.shape[1]), dtype=np.float32)
    for i in range(0, 40):
        s += np.dot(z[i].T, z[i])

    _, eigenvectors = np.linalg.eigh(np.dot(np.linalg.inv(s), Sb))
    eigenvectors = np.flip(eigenvectors)
    u_reduced = eigenvectors[:, :39]
    d_train_reduced = np.dot(d_train, u_reduced)
    d_test_reduced = np.dot(d_test, u_reduced)
    print("******************* LDA *********************")
    knn(d_train_reduced, d_test_reduced, y_train, y_test, 0, 0)


def pca(d_train, y_train, d_test, y_test):
    alphas = [0.8, 0.85, 0.9, 0.95]
    alphas_vectors = [0, 0, 0, 0]

    mean = np.mean(d_train, 0)
    # print("\n mean matrix\n", mean)

    centered = d_train - mean
    # print("\n centered  matrix\n", centered)

    # covarianceMatrix = np.dot(centered.T, centered) * (1 / d.shape[0])
    covarianceMatrix = np.cov(centered, rowvar=False, bias=True)
    # print("\nCovariance Matrix \n",covarianceMatrix)
    # print(covarianceMatrix.shape)

    eigen_value, eigen_vectors = np.linalg.eigh(covarianceMatrix)
    # print("\n eigen value  matrix\n", eigen_value.shape)
    # print("\n eigen vectors  matrix\n", eigen_vectors.shape)
    # x = np.transpose(np.dot( d,np.transpose(d2)))
    # print("\n projected  matrix\n", x)
    eigen_vectors = eigen_vectors.T[::-1]
    eigen_value = eigen_value.T[::-1]
    eigen_sum = eigen_value.sum()
    sum_eigen = 0
    i = 0
    j = 0
    for x in eigen_value:
        sum_eigen += x
        if sum_eigen / eigen_sum >= alphas[i]:
            alphas_vectors[i] = j
            i += 1
            if i == 4:
                break
        j += 1
    # print(alphas_vectors)
    i = 0
    print("******************* PCA *********************")
    for alphas_vector in alphas_vectors:
        U = eigen_vectors[0:alphas_vector]
        U = U[::-1]
        A_train = np.dot(d_train, np.transpose(U))
        A_test = np.dot(d_test, np.transpose(U))
        r_accuracies = []
        knn(A_train, A_test, y_train, y_test, alphas[i], alphas_vector)
        i += 1


def read():  # reading the pgm files and creating the data matrix
    for i in range(1, 41):
        for filepath in glob.glob(
                os.path.join(f'faces/s{i}', '*.pgm')):
            with open(filepath, 'rb') as f:
                image = plt.imread(f).flatten()
            image = list(image)
            D.append(image)
    global y
    for i in range(1, 41):  # creating the y label vector
        for j in range(1, 11):
            y.append(i)
    y = np.array([y]).T


def split(size):
    d_train = []
    y_train = []
    d_test = []
    y_test = []
    for i in range(1, size, 2):
        d_train.append(list(D[i]))
        y_train.append(list(y[i]))
    for i in range(0, size, 2):
        d_test.append(list(D[i]))
        y_test.append(list(y[i]))

    d_train = np.array(d_train)
    y_train = np.array(y_train)
    d_test = np.array(d_test)
    y_test = np.array(y_test)
    return d_train, y_train, d_test, y_test


def split_bonus(size):
    tr = [0, 1, 2, 3, 4, 5, 6]
    ts = [7, 8, 9]
    d_train = []
    y_train = []
    d_test = []
    y_test = []
    for j in range(0, size, 10):
        for i in tr:
            d_train.append(list(D[i + j][:]))
            y_train.append(list(y[i + j][:]))
    for i in range(0, size, 10):
        for k in ts:
            d_test.append(list(D[i + k][:]))
            y_test.append(list(y[i + k][:]))

    d_train = np.array(d_train)
    y_train = np.array(y_train)
    d_test = np.array(d_test)
    y_test = np.array(y_test)
    print(d_train.shape)
    print(d_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    return d_train, y_train, d_test, y_test


def faces_nonFaces():
    global y
    for filepath in glob.glob(
            os.path.join('non-faces/', '*.pgm')):
        with open(filepath, 'rb') as f:
            image = plt.imread(f).flatten()
            image = list(image)
            D.append(image)

    y = []
    for i in range(400):  # creating the y label vector
        y.append(1)
    for i in range(140):
        y.append(0)
    y = np.array([y]).T
    d_train, y_train, d_test, y_test = split(540)
    pca(d_train, y_train, d_test, y_test)
    # lda(d_train,y_train,d_test,y_test)


if __name__ == '__main__':
    # d_train, y_train, d_test, y_test = split(400)
    # print(y_train)
    # pca(d_train,y_train,d_test,y_test)
    # lda(d_train,y_train,d_test,y_test)
    d_train, y_train, d_test, y_test = split_bonus(400)
    # print(y_train)
    # pca(d_train,y_train,d_test,y_test)
    lda(d_train, y_train, d_test, y_test)
    # faces_nonFaces()
