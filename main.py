import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# reading the pgm files and creating the data matrix
D = []
for i in range(1, 41):
    for filepath in glob.glob(
            os.path.join('faces/s{}'.format(i), '*.pgm')):
        with open(filepath, 'rb') as f:
            image = plt.imread(f).flatten()
        image = list(image)
        D.append(image)
# print(D)
# creating the y label vector
y = []
for i in range(1, 41):
    for j in range(1, 11):
        y.append(i)
y = np.array([y]).T
#print(y)
d_train = []
y_train = []
d_test = []
y_test = []
for i in range(1, 401, 2):
    d_train.append(list(D[i][:]))
    y_train.append(list(y[i][:]))
for i in range(0, 400, 2):
    d_test.append(list(D[i][:]))
    y_test.append(list(y[i][:]))


