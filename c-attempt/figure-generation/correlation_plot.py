import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from parula import parula
import os

cm = 1/2.54

path = os.path.dirname(__file__)

matplotlib.rcParams['text.usetex'] = True

p02s = [1.,0.3617]

colors = ['magenta','cyan']

for p02s_idx in range(len(p02s)):
    P_1 = 1.
    P_2 = p02s[p02s_idx]
    alpha = P_2 / P_1

    gamma_tau = np.linspace(-5,5,1000)

    g_2_2 = (2 * P_1 * P_2 + P_1**2 * (1-np.exp(-np.abs(gamma_tau))) + P_2**2 * (1-np.exp(-np.abs(gamma_tau)))) / ((P_1 + P_2)**2)
    
    plt.plot(gamma_tau, g_2_2, c=colors[p02s_idx], linewidth=0.75, label=r'$\alpha=' + f'{alpha}' + '$')

# plt.title(r'$\alpha=' + str(alpha) + r'$', fontsize=20)
plt.legend(fontsize=10, ncols=2)
plt.xlim(-5,5)
plt.xticks(fontsize=10)
plt.xlabel(r'$\mathrm{Normalized\;Time\;Lag}\;\tau\Gamma$', fontsize=10)
plt.ylim(0.,1.0)
plt.yticks([0.,0.5,1.0], fontsize=10)
plt.ylabel(r'$g_{2}^{(2)}$', fontsize=10)
plt.gca().set_aspect(10 / (3 * 1.0))
plt.tight_layout()
plt.gcf().set_figwidth(val=0.75 * 15.3978 * cm)
plt.savefig(os.path.join(path,'coincidence.png'), dpi=600, bbox_inches='tight')
plt.close()