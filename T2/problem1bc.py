import numpy as np
import scipy.stats
import scipy.integrate
from matplotlib import pyplot as plt

def chi_density(x, D):
    return 2*x*scipy.stats.chi2.pdf(x**2, D)

x = np.linspace(0.1, 14, 1000)
D_list = [1, 5, 10, 20, 40, 60, 80, 100]
for i in D_list:
    plt.plot(x, chi_density(x, i))
    plt.xlim([0, 14])
    plt.ylim([0, 0.8])
    plt.title('Chi Distributions with Different Degrees of Freedom')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend('top right', labels = ['D = ' + str(i) for i in D_list])
plt.show()

def chi_CDF(D, lower_end_point, upper_end_point):
    return scipy.integrate.quad(chi_density, lower_end_point, upper_end_point, args = (D))

x = np.linspace(0.1, 14, 1000)
D = 100
cdf_vals = np.zeros(len(x))
for i in range(len(x)):
    cdf_vals[i] = chi_CDF(D, 0.1, x[i])[0]

plt.plot(x, cdf_vals)
plt.xlim([0, 14])
plt.ylim([0, 1.01])
plt.title('Chi CDF when D = 100')
plt.xlabel('x')
plt.ylabel('P(X < x)')
plt.show()

