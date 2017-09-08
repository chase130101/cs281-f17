import numpy as np
from matplotlib import pyplot as plt

def f(x):
    return np.cos(x) + x**2 + np.exp(x)

def grad_f(x):
    return -np.sin(x) + 2*x + np.exp(x)

def grad_check(x, epsilon):
    return (f(x + epsilon) - f(x - epsilon)) / (2*epsilon)


x_vals = [0, 2, 5, 10]
x_lists = []
for x in x_vals:
    x_lists.append([])
    for i in range(1, 10):
        x_lists[-1].append(grad_check(x, 1.5**(-i)))

for i in range(4):
    plt.plot(range(-1, -10, -1), x_lists[i], 'bo')
    plt.plot(range(-1, -10, -1), [grad_f(x_vals[i]) for j in range(1, 10)], 'r')
    plt.ylim([x_lists[i][0] - 2*(x_lists[i][0] - grad_f(x_vals[i])), x_lists[i][0]])
    plt.xlabel('Epsilon (log base 1.5)')
    plt.ylabel('Value / numerical estimate of gradient')
    plt.legend(labels = ['Numerical estimate of gradient', 'Value of gradient'])
    plt.title('Numerically estimating the gradient at x = ' + str(x_vals[i]))
    plt.show()