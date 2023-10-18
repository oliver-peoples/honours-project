import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from parula import parula

matplotlib.rcParams['text.usetex'] = True

p02s = [1.,0.3167]

colors = ['magenta','cyan']

for p02s_idx in range(len(p02s)):
    P_1 = 1.
    P_2 = p02s[p02s_idx]
    alpha = P_2 / P_1

    gamma_tau = np.linspace(-5,5,1000)

    g_2_2 = (2 * P_1 * P_2 + P_1**2 * (1-np.exp(-np.abs(gamma_tau))) + P_2**2 * (1-np.exp(-np.abs(gamma_tau)))) / ((P_1 + P_2)**2)
    
    plt.plot(gamma_tau, g_2_2, c=colors[p02s_idx], label=r'$\alpha=' + f'{alpha:.3f}' + '$')

plt.title(r'$\alpha=' + str(alpha) + r'$', fontsize=20)
plt.legend(fontsize=12, cols=2)
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