import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from parula import parula

cm = 1/2.54

matplotlib.rcParams['text.usetex'] = True

alpha = np.linspace(0,10,1000)

g_2_2 = (2 * alpha) / (1 + alpha)**2

# plt.title(r'$\alpha=' + str(alpha) + r'$', fontsize=20)
plt.plot(alpha, g_2_2, 'k')
plt.xlim(0,10)
plt.xticks(fontsize=12)
plt.xlabel(r'$\alpha=P_{2}/P_{1}$', fontsize=12)
plt.ylim(0.,0.55)
plt.yticks([0.,0.5], fontsize=12)
plt.ylabel(r'$g_{2}^{(2)}$', fontsize=12)
plt.gca().set_aspect(10/(3*0.55))
plt.tight_layout()
plt.gcf().set_figwidth(val=0.99 * 15.3978 * cm)
plt.savefig('alpha.png', dpi=600, bbox_inches='tight')
plt.close()