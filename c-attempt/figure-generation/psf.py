import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import jv
# from parula import parula
import os

cm = 1/2.54

path = os.path.dirname(__file__)

matplotlib.rcParams['text.usetex'] = True

rho_linspace = np.linspace(-2,2,1000)

airy = (jv(1.,2*np.pi * rho_linspace) / (2*np.pi * rho_linspace))**2
gauss = 0.25 * np.exp(-rho_linspace**2 /(2*0.21**2))

plt.plot(rho_linspace,airy)
plt.plot(rho_linspace,gauss)
plt.show()