import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from matplotlib import rc

rc('axes', linewidth=1, labelsize=12)
rc('font', family='Times New Roman')

data = np.loadtxt('data/Dbig.txt')
index = np.arange(data.shape[0])
np.random.shuffle(index)
testset = data[index[8192:]]
test_X = testset[:, :2]
test_y = testset[:, 2]

trainsizes = [32, 128, 512, 2048, 8192]
errs = []
print('trainsize', '# nodes', 'err')
for trainsize in trainsizes:
    trainset = data[index[:trainsize]]

    train_X = trainset[:, :2]
    train_y = trainset[:, 2]

    clf = DecisionTreeClassifier()
    clf = clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)
    err = (pred_y != test_y).sum() / test_y.shape[0]
    print(trainsize, clf.tree_.capacity, np.round(err, 6))
    errs.append(err)
plt.plot(trainsizes, errs, 'o-', markersize=5)
plt.xlabel('# train samples')
plt.ylabel('err')
plt.title('$n$ vs $err_n$')
plt.show()