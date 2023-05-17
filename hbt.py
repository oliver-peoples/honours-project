import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['text.usetex'] = True

alpha_linspace = np.linspace(0,1500,2000)
g2_linspace = (2 * alpha_linspace) / (1 + alpha_linspace)**2

plt.plot(alpha_linspace, g2_linspace, color='b')
plt.xlabel(r'$\alpha$', fontsize=16)
plt.ylabel(r'$g_{2}^{(2)}$', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.xlim(0,15)
# plt.ylim(0,0.525)
# plt.gca().set_aspect(15 / (2.5*0.525))
# plt.tight_layout()
plt.savefig('quantum_correlation_alpha.png', dpi=600, bbox_inches="tight")