import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from matplotlib import rc
from sklearn.metrics import roc_curve, roc_auc_score

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

rc('axes', linewidth=1, labelsize=12)
rc('font', family='Times New Roman')


def loggraddescent(X, y, eta, w_init, it):
    W = torch.zeros((w_init.shape[0], it + 1)).cuda()
    y = y.reshape(-1, 1)
    W[:, [0]] = w_init
    for k in range(it):
        grad = X.T @ (1 / (1 + torch.exp(-X @ W[:, [k]])) - y)
        W[:, [k + 1]] = W[:, [k]] - eta * grad
    return W


data = pd.read_csv('data/emails.csv')

n = 5000  # num of samples
p = 3000  # num of feature
X = data.values[:, 1:1+p].astype('float')
y = data.values[:, -1].astype('float')

train_X = torch.Tensor(X[:4000]).cuda()
train_y = y[:4000]
test_X = torch.Tensor(X[4000:]).cuda()
test_y = y[4000:]
n_test = test_X.shape[0]

# kNN
dis_matrix = []
for j in tqdm(range(n_test)):
    dis = torch.norm((test_X[j] - train_X), dim=1)  # todo
    dis_matrix.append(np.array(dis.cpu()))
dis_matrix = np.array(dis_matrix)
index = np.argsort(dis_matrix, axis=1)[:, :5]
pred_test_y, freq = stats.mode(train_y[index], axis=1)
# pred_test_y = np.round(train_y[index].mean(1))
y_pred_1 = pred_test_y.flatten()
y_prob_1 = (train_y[index] == 1).sum(1) / 5

# logistic regression
eta = 1e-3
it = 100
w_init = torch.randn((p, 1)).cuda()
train_X = torch.Tensor((train_X.cpu().numpy() - X.mean(0)) / X.std(0)).cuda()  # standarize
test_X = torch.Tensor((test_X.cpu().numpy() - X.mean(0)) / X.std(0)).cuda()  # standarize
W = loggraddescent(train_X, torch.Tensor(train_y).cuda(), eta=eta, w_init=w_init, it=it)
pred_y = (test_X @ W[:, [-1]]).cpu().numpy().flatten()
pred_test_y = np.sign(pred_y)
pred_test_y[pred_test_y == -1] = 0
y_prob_2 = 1/(1+np.exp(-pred_y))
y_pred_2 = pred_test_y

fig, ax = plt.subplots(figsize=(6, 4))
fpr1, tpr1, _ = roc_curve(test_y, y_prob_1)
fpr2, tpr2, _ = roc_curve(test_y, y_prob_2)
auc1 = roc_auc_score(test_y, y_prob_1)
auc2 = roc_auc_score(test_y, y_prob_2)

plt.plot(fpr1, tpr1, label=f'KNeighborsClassifier (AUC={auc1:.2f})')
plt.plot(fpr2, tpr2, label=f'LogisticRegression (AUC={auc2:.2f})')

# ax.set_aspect('equal')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
# plt.xticks(ks)
# # plt.yticks([-2, 0, 2])
# plt.title('kNN 5-fold Cross Validation')
plt.grid()
plt.tight_layout()

plt.show()