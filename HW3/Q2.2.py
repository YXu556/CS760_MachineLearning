import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import rc


rc('axes', linewidth=1, labelsize=12)
rc('font', family='Times New Roman')

data = pd.read_csv('data/emails.csv')

n = 5000  # num of samples
p = 3000  # num of feature
X = data.values[:, 1:1+p].astype('float')
y = data.values[:, -1].astype('float')

indices = np.arange(n)
grouped_id = indices.reshape(5, 1000)

# Q2
for i in range(5):
    print(f'========== CV - {i+1} ============')
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
    # out of memory
    # dis_matrix = np.linalg.norm(test_X.reshape(n_test, 1, p) - train_X.reshape(1, *train_X.shape), axis=2)
    index = np.argmin(dis_matrix, axis=1)
    pred_test_y = train_y[index.reshape(n_test, 1)].flatten()

    y_true = test_y
    y_pred = pred_test_y
    accuracy = (y_true == y_pred).sum() / y_true.shape[0]
    precision = (y_pred[y_true == 1] == 1).sum() / (y_pred == 1).sum()
    recall = (y_pred[y_true == 1] == 1).sum() / (y_true == 1).sum()

    print(f'Accuracy = {accuracy:.4f}\nPrecision = {precision:.4f}\nRecall = {recall:.4f}')
    # print(f'Fold {i+1} & {accuracy:.4f} & {precision:.4f} & {recall:.4f}')