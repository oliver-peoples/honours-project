import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from parula import parula
import os

cm = 1/2.54

path = os.path.dirname(__file__)

matplotlib.rcParams['text.usetex'] = True

alpha = np.linspace(0,10,1000)

g_2_2 = (2 * alpha) / (1 + alpha)**2

# plt.title(r'$\alpha=' + str(alpha) + r'$', fontsize=20)
plt.plot(alpha, g_2_2, c='magenta', linewidth=0.75)
plt.xlim(0,10)
plt.xticks(fontsize=10)
plt.xlabel(r'$\alpha=P_{2}/P_{1}$', fontsize=10)
plt.ylim(0.,0.55)
plt.yticks([0.,0.5], fontsize=10)
plt.ylabel(r'$g_{2}^{(2)}$', fontsize=10)
plt.gca().set_aspect(10/(3*0.55))
plt.tight_layout()
plt.gcf().set_figwidth(val=0.75 * 15.3978 * cm)
plt.savefig(os.path.join(path,'alpha.png'), dpi=600, bbox_inches='tight')
plt.close()