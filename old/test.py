import numpy as np
import matplotlib.pyplot as plt

s = np.random.poisson(50, 1000000)

import matplotlib.pyplot as plt

count, bins, ignored = plt.hist(s, 25, density=True)

plt.show()