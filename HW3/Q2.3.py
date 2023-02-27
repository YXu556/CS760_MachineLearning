import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import rc

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
X = data.values[:, 1:1 + p].astype('float')
X = (X - X.mean(0)) / X.std(0)  # standarize
y = data.values[:, -1].astype('float')

indices = np.arange(n)
grouped_id = indices.reshape(5, 1000)
eta = 1e-3
it = 100


for i in range(5):
    print(f'========== CV - {i + 1} ============')
    train_X = torch.Tensor(X[np.delete(indices, grouped_id[i])]).cuda()
    train_y = torch.Tensor(y[np.delete(indices, grouped_id[i])]).cuda()
    test_X = torch.Tensor(X[grouped_id[i]]).cuda()
    test_y = y[grouped_id[i]]

    w_init = torch.randn((p, 1)).cuda()

    # W = loggraddescent(np.array(train_X), np.array(train_y), eta=eta, w_init=np.array(w_init), it=20)
    W = loggraddescent(train_X, train_y, eta=eta, w_init=w_init, it=it)
    pred_test_y = np.sign((test_X @ W[:, [-1]]).cpu().numpy().flatten())
    pred_test_y[pred_test_y == -1] = 0

    y_true = test_y
    y_pred = pred_test_y
    accuracy = (y_true == y_pred).sum() / y_true.shape[0]
    precision = (y_pred[y_true == 1] == 1).sum() / (y_pred == 1).sum()
    recall = (y_pred[y_true == 1] == 1).sum() / (y_true == 1).sum()

    # print(f'Accuracy = {accuracy:.4f}\nPrecision = {precision:.4f}\nRecall = {recall:.4f}')
    print(f'Fold {i+1} & {accuracy:.4f} & {precision:.4f} & {recall:.4f}\\\\')
