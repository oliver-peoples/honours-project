import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from parula import parula

matplotlib.rcParams['text.usetex'] = True

alpha = np.linspace(0,10,1000)

g_2_2 = (2 * alpha) / (1 + alpha)**2

# plt.title(r'$\alpha=' + str(alpha) + r'$', fontsize=20)
plt.plot(alpha, g_2_2, 'k')
plt.xlim(0,10)
plt.xticks(fontsize=16)
plt.xlabel(r'$\alpha=P_{2}/P_{1}$', fontsize=18)
plt.ylim(0.,0.55)
plt.yticks([0.,0.5], fontsize=16)
plt.ylabel(r'$g_{2}^{(2)}$', fontsize=18)
plt.gca().set_aspect(8)
plt.tight_layout()
plt.savefig('alpha.png', dpi=600, bbox_inches='tight')
plt.close()