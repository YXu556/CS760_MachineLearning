import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from matplotlib import rc

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

rc('axes', linewidth=1, labelsize=12)
rc('font', family='Times New Roman')

data = pd.read_csv('data/emails.csv')

n = 5000  # num of samples
p = 3000  # num of feature
X = data.values[:, 1:1+p].astype('float')
y = data.values[:, -1].astype('float')

indices = np.arange(n)
grouped_id = indices.reshape(5, 1000)

accs = []
ks = [1, 3, 5, 7, 10]
for k in ks:
    print(f'========== K = {k} ============')
    y_pred = np.array([])
    # acc = []
    for i in range(5):
        train_X = torch.Tensor(X[np.delete(indices, grouped_id[i])]).cuda()
        train_y = y[np.delete(indices, grouped_id[i])]
        test_X = torch.Tensor(X[grouped_id[i]]).cuda()
        test_y = y[grouped_id[i]]
        n_test = test_X.shape[0]

        dis_matrix = []
        for j in tqdm(range(n_test)):
            dis = torch.norm((test_X[j] - train_X), dim=1)  # todo
            dis_matrix.append(np.array(dis.cpu()))
        dis_matrix = np.array(dis_matrix)
        index = np.argsort(dis_matrix, axis=1)[:, :k]
        pred_test_y, _ = stats.mode(train_y[index], axis=1)
        # pred_test_y = np.round(train_y[index].mean(1))
        y_pred = np.hstack([y_pred, pred_test_y.flatten()])

        # y_true = test_y
        # y_pred = pred_test_y
        # accuracy = (y_true == y_pred).sum() / y_true.shape[0]
        # acc.append(accuracy)
    y_true = y
    y_pred = y_pred
    accuracy = (y_true == y_pred).sum() / y_true.shape[0]
    precision = (y_pred[y_true == 1] == 1).sum() / (y_pred == 1).sum()
    recall = (y_pred[y_true == 1] == 1).sum() / (y_true == 1).sum()
    # accuracy = np.mean(acc)
    # print(f'Accuracy = {accuracy:.4f}\nPrecision = {precision:.4f}\nRecall = {recall:.4f}')
    print(f'{k} & {accuracy:.4f}\\\\')
    accs.append(accuracy)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ks, accs, 'o-')
# ax.set_aspect('equal')
plt.xlabel('k')
plt.ylabel('Average accuracy')
# plt.legend(loc=1)
plt.xticks(ks)
# plt.yticks([-2, 0, 2])
plt.title('kNN 5-fold Cross Validation')
plt.grid()
plt.tight_layout()

plt.show()