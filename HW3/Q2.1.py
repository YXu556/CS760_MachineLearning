import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


rc('axes', linewidth=1, labelsize=12)
rc('font', family='Times New Roman')


def scatterplot(X, y, marker='o', alpha=0.5, label_prefix='train'):
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    plt.scatter(X_0[:, 0], X_0[:, 1], s=5, marker=marker, color='b', alpha=alpha, label=label_prefix+'_0')
    plt.scatter(X_1[:, 0], X_1[:, 1], s=5, marker=marker, color='r', alpha=alpha, label=label_prefix+'_1')


data = np.loadtxt('data/D2z.txt')

p = 2  # num of feature
train_X = data[:, :p]
train_y = data[:, p]

test_data = np.meshgrid(np.arange(-2,2.1,0.1), np.arange(-2,2.1,0.1))
test_X = np.hstack([test_data[0].reshape(-1, 1), test_data[1].reshape(-1, 1)])
n_test = test_X.shape[0]

dis_matrix = np.linalg.norm(test_X.reshape(n_test, 1, p)-train_X.reshape(1, *train_X.shape), axis=2)
index = np.argmin(dis_matrix, axis=1)
test_y = train_y[index.reshape(n_test, 1)].flatten()

# plot
fig, ax = plt.subplots(figsize=(4, 4))
scatterplot(train_X, train_y)
scatterplot(test_X, test_y, marker='^', alpha=0.2, label_prefix='pred')
ax.set_aspect('equal')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc=1)
plt.xticks([-2, 0, 2])
plt.yticks([-2, 0, 2])
plt.tight_layout()

plt.show()