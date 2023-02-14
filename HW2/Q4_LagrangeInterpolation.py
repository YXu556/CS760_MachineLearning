import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from matplotlib import rc
from numpy.polynomial.polynomial import Polynomial
from pathlib import Path

rc('axes', linewidth=1, labelsize=12)
rc('font', family='Times New Roman')


def error(y_pred, y_true):
    return np.sqrt(((y_pred-y_true)**2).mean())


X_train = np.linspace(0, np.pi, 10)
y_train = np.sin(X_train)
X_test = np.linspace(0, np.pi, 100)
y_test = np.sin(X_test)

# train
poly = lagrange(X_train, y_train)

# pred
pred_y_train = Polynomial(poly.coef[::-1])(X_train)
pred_y_test = Polynomial(poly.coef[::-1])(X_test)

err_train = error(pred_y_train, y_train)
err_test = error(pred_y_test, y_test)

print(f'train_error={err_train:.3e}, test_error={err_test:.3e}')

plt.figure(figsize=(5, 4))
plt.plot(X_train, y_train, label='train_origin')
plt.plot(X_train, pred_y_train, label=f'train_pred')
plt.plot(X_test, pred_y_test, label=f'test_pred')
plt.xlabel('x')
plt.ylabel('y')
plt.title('w/ noise')
plt.tight_layout()
plt.legend()
plt.show()


for sigma in np.logspace(-2, 0, 4):
    X_train_noise = X_train + np.random.normal(0, sigma, 10)
    y_train_noise = np.sin(X_train_noise)

    poly = lagrange(X_train_noise, y_train)

    pred_y_train = Polynomial(poly.coef[::-1])(X_train_noise)
    pred_y_test = Polynomial(poly.coef[::-1])(X_test)

    err_train = error(pred_y_train, y_train)
    err_test = error(pred_y_test, y_test)
    print(f'sigma={sigma: .3f}, train_error={err_train:.3e}, test_error={err_test:.3e}')

    order_X = np.argsort(X_train_noise)
    plt.figure(figsize=(5, 4))
    plt.plot(X_train_noise[order_X], y_train[order_X], label='origin')
    plt.plot(X_train_noise[order_X], pred_y_train[order_X], label='train-pred')
    plt.plot(X_test, pred_y_test, label='test-pred')
    plt.title(f"sigma={sigma:.3f}")
    plt.legend()
    plt.show()
