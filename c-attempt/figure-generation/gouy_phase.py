import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from parula import parula
import os

cm = 1/2.54

path = os.path.dirname(__file__)

matplotlib.rcParams['text.usetex'] = True

z_b = np.linspace(-20,20,1000)

phi_z = np.arctan(z_b)

plt.plot(z_b, phi_z, c='magenta', linewidth=0.75)
plt.xlabel(r'$z/b$', fontsize=10)
plt.xlim(z_b[0],z_b[-1])
plt.xticks(fontsize=10)
plt.ylim(-np.pi/2,np.pi/2)
plt.yticks([-np.pi/2,0,np.pi/2],[r'$-\pi/2$', r'$0$', r'$\pi/2$'], fontsize=10)
plt.ylabel(r'$\phi\left(z\right)$', fontsize=10)
plt.gca().set_aspect((z_b[-1] - z_b[0]) / (3 * np.pi))
plt.tight_layout()
plt.gcf().set_figwidth(val=0.75 * 15.3978 * cm)
plt.savefig(os.path.join(path,'gouy_phase_shift.png'), dpi=600, bbox_inches='tight')
plt.close()