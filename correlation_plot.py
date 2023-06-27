import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from parula import parula

matplotlib.rcParams['text.usetex'] = True

P_1 = 1.
P_2 = 1.
alpha = P_2 / P_1

gamma_tau = np.linspace(-5,5,1000)

g_2_2 = (2 * P_1 * P_2 + P_1**2 * (1-np.exp(-np.abs(gamma_tau))) + P_2**2 * (1-np.exp(-np.abs(gamma_tau)))) / ((P_1 + P_2)**2)

plt.title(r'$\alpha=' + str(alpha) + r'$', fontsize=20)
plt.plot(gamma_tau, g_2_2, 'k')
plt.xlim(-5,5)
plt.xticks(fontsize=16)
plt.xlabel(r'$\mathrm{Normalized\;Time\;Lag}\;\tau\Gamma$', fontsize=18)
plt.ylim(0.,1.0)
plt.yticks([0.,0.5,1.0], fontsize=16)
plt.ylabel(r'$g_{2}^{(2)}$', fontsize=18)
plt.gca().set_aspect(4)
plt.tight_layout()
plt.savefig('coincidence.png', dpi=600, bbox_inches='tight')
plt.close()