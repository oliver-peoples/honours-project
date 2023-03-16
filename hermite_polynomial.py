import numpy as np
from numpy.polynomial.hermite import Hermite

n = 4

H = Hermite([0] * (n + 1), domain=[-np.inf, np.inf], window=[-np.inf, np.inf], norm='physicist')

x = 1.5
result = H(x)

print("The physicist's Hermite polynomial of order {} at x = {} is: {}".format(n, x, result))