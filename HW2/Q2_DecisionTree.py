import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from matplotlib import rc

rc('axes', linewidth=1, labelsize=12)
rc('font', family='Times New Roman')
node_id = 0


class Node:
    def __init__(self, featureindex, value):
        self.featureindex = featureindex
        self.value = value
        self.gainratio = 0
        self.left_child = None
        self.right_child = None
        self.is_leaf = False
        self.label = None
        self.index = None


def scatterplot(X, y):
    fig, ax = plt.subplots(figsize=(5, 5))
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    plt.scatter(X_0[:, 0], X_0[:, 1], s=5, label='0')
    plt.scatter(X_1[:, 0], X_1[:, 1], s=5, label='1')
    ax.set_aspect('equal')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend(loc=1)
    # plt.tight_layout()
    # plt.show()


def DetermineCandidateSplits(X, y):
    C = {}
    for i in range(2):
        v = X[:, i]
        order = np.argsort(v)
        v_order = v[order]
        y_order = y[order]
        pairs = y_order[:-1] != y_order[1:]
        candidate_tmp = np.unique(v_order[1:][pairs])
        C[i] = candidate_tmp
    return C


def CalConditionEntropy(y):
    e = 1e-8
    n = y.shape[0]
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    p0, p1 = n0 / n, n1 / n
    if p0 == 0:
        p0_log = p0 + e
    else:
        p0_log = p0
    if p1 == 0:
        p1_log = p1 + e
    else:
        p1_log = p1
    return -p0 * np.log2(p0_log) - p1 * np.log2(p1_log)


def FindBestSplit(X, y, C, verbose_all=False, verbose=True):
    best_gain_ratio = -np.inf
    best_i, best_c = 0, 0
    h_y = CalConditionEntropy(y)
    for i in range(2):
        # print(f' =============== x_{i} ================ ')
        for c in C[i]:
            s = X[:, i] >= c

            n = y.shape[0]
            n0 = s.sum()
            n1 = n - n0
            p0, p1 = n0 / n, n1 / n

            if n0 == 0 or n1 == 0:
                continue

            y0, y1 = y[s], y[~s]
            h0 = CalConditionEntropy(y0)
            h1 = CalConditionEntropy(y1)

            info_gain = h_y - (p0 * h0 + p1 * h1)
            h_s = -p0 * np.log2(p0) - p1 * np.log2(p1)

            gain_ratio = info_gain / h_s

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_i = i
                best_c = c
            if verbose_all:
                print(f'x_{i+1} >= {c}: gain_ratio = {gain_ratio}, info_gain = {info_gain}')

    if best_gain_ratio > 0:
        if verbose:
            print(f'When x_{best_i+1} >= {best_c}, the best gain ratio = {best_gain_ratio}')
        return best_i, best_c, best_gain_ratio
    else:
        if verbose:
            print(f'Leaf node, class', int(mode(y)[0][0]))
        return mode(y)[0][0]


def MakeSubtree(X, y, parent=None, p_label=None, verbose=True):
    global node_id
    node_id += 1
    if verbose:
        if p_label is None:
            print(f' =================== root =================== ')
            print(f"Node {node_id}：", end='\t')
        else:
            print(f' ============ None {parent} - {p_label} child ============ ')
            print(f"Node {node_id}：", end='\t')

    C = DetermineCandidateSplits(X, y)
    best = FindBestSplit(X, y, C, verbose=verbose)
    if type(best) is not tuple:
        node = Node(-1, -1)
        node.is_leaf = True
        node.label = best
        node.index = node_id
        return node
    else:
        i, c, gr = best
        node = Node(i, c)
        node.gainratio = gr
        s = X[:, i] >= c

        X0, y0 = X[s], y[s]
        X1, y1 = X[~s], y[~s]
        node.index = node_id
        node.left_child = MakeSubtree(X0, y0, parent=node.index, p_label='left', verbose=verbose)
        node.right_child = MakeSubtree(X1, y1, parent=node.index, p_label='right', verbose=verbose)
        return node


def DisplayDecisionSurf(node, min1=0, max1=1, min2=0, max2=1, X=None, y=None, n=1000):
    xx1, xx2 = np.meshgrid(np.linspace(min1, max1, n), np.linspace(min2, max2, n))
    yy = []
    for data in zip(xx1.flatten(), xx2.flatten()):
        pred = predict(node, data)
        yy.append(pred)

    yy = np.array(yy).reshape(n, n)
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.contourf(xx1, xx2, yy, cmap='Paired')

    if X is not None:
        X_0 = X[y == 0]
        X_1 = X[y == 1]
        plt.scatter(X_0[:, 0], X_0[:, 1], s=5, label='0')
        plt.scatter(X_1[:, 0], X_1[:, 1], s=5, label='1')
        plt.legend(loc=1)
    ax.set_aspect('equal')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    # plt.tight_layout()


def predict(node, data):
    if node.is_leaf:
        return node.label
    if data[node.featureindex] >= node.value:
        return predict(node.left_child, data)
    else:
        return predict(node.right_child, data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question_no', type=int, required=True,
                        help='Select from [3, 4, 51, 52, 6, 7]')
    args = parser.parse_args()

    question_no = args.question_no

    if question_no == 3:
        data = np.loadtxt('data/Druns.txt')
        X = data[:, :2]
        y = data[:, 2]
        C = DetermineCandidateSplits(X, y)
        best = FindBestSplit(X, y, C, verbose_all=True)
    elif question_no == 4:
        data = np.loadtxt('data/D3leaves.txt')
        X = data[:, :2]
        y = data[:, 2]
        MakeSubtree(X, y)
    elif question_no == 51:
        data = np.loadtxt('data/D1.txt')
        X = data[:, :2]
        y = data[:, 2]
        MakeSubtree(X, y)
    elif question_no == 52:
        data = np.loadtxt('data/D2.txt')
        X = data[:, :2]
        y = data[:, 2]
        MakeSubtree(X, y)
    elif question_no == 6:
        data = np.loadtxt('data/D1.txt')
        X = data[:, :2]
        y = data[:, 2]
        scatterplot(X, y)
        plt.title('D1 scatter plot')
        plt.show()

        node = MakeSubtree(X, y, verbose=False)
        DisplayDecisionSurf(node, X=X, y=y)
        plt.title('D1 decision area')
        plt.show()

        data = np.loadtxt('data/D2.txt')
        X = data[:, :2]
        y = data[:, 2]
        scatterplot(X, y)
        plt.title('D2 scatter plot')
        plt.show()

        node = MakeSubtree(X, y, verbose=False)
        DisplayDecisionSurf(node, X=X, y=y)
        plt.title('D2 decision area')
        plt.show()

    elif question_no == 7:
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
            node_id = 0
            trainset = data[index[:trainsize]]

            train_X = trainset[:, :2]
            train_y = trainset[:, 2]

            node = MakeSubtree(train_X, train_y, verbose=False)
            pred_y = []
            for tmp_X in test_X:
               pred_y.append(predict(node, tmp_X))
            err = (pred_y != test_y).sum()/test_y.shape[0]
            print(trainsize, node_id, np.round(err, 6))
            errs.append(err)

            DisplayDecisionSurf(node, min1=data[:, 0].min(), max1=data[:, 0].max(),
                                min2=data[:, 1].min(), max2=data[:, 1].max(),)
            plt.title(f'n={trainsize}')
            plt.show()

        plt.plot(trainsizes, errs, 'o-', markersize=5)
        plt.xlabel('# train samples')
        plt.ylabel('err')
        plt.title('$n$ vs $err_n$')
        plt.show()
