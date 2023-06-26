import numpy as np
from numpy.polynomial.hermite import Hermite

# Set the order of the physicist's Hermite polynomial
n = 4

# Create the Hermite object with physicist's normalization
H = Hermite([0] * (n + 1), domain=[-np.inf, np.inf], window=[-np.inf, np.inf])

# Evaluate the physicist's Hermite polynomial at x = 1.5
x = 1.5
result = H(x)

# Print the result
print("The physicist's Hermite polynomial of order {} at x = {} is: {}".format(n, x, result))