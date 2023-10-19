import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from parula import parula
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

cm = 1/2.54

path = os.path.dirname(__file__)

matplotlib.rcParams['text.usetex'] = True

P_1 = 1.
P_2 = 0.78

scale = 0.5

pm = 3

e_1_xy = scale * np.array([-0.5,-0.25])
e_2_xy = scale * np.array([0.5,0.25])

r = np.linalg.norm(e_1_xy - e_2_xy)

d_w = 1.

print(r/d_w)

x_linspace = d_w * np.linspace(-pm,pm,1000)
y_linspace = d_w * np.linspace(pm,-pm,1000)

x_meshgrid,y_meshgrid = np.meshgrid(x_linspace,y_linspace)

e_1_r_meshgrid = np.sqrt((x_meshgrid - e_1_xy[0])**2 + (y_meshgrid - e_1_xy[1])**2)
e_2_r_meshgrid = np.sqrt((x_meshgrid - e_2_xy[0])**2 + (y_meshgrid - e_2_xy[1])**2)

p_1_meshgrid = P_1 * np.exp(-(e_1_r_meshgrid**2)/(2*d_w**2))
p_2_meshgrid = P_2 * np.exp(-(e_2_r_meshgrid**2)/(2*d_w**2))

g1 = (p_1_meshgrid + p_2_meshgrid) / (P_1 + P_2)

plt.figure()
plt.title(r'$r=' + f'{r:.3f}' + r'\sigma$', fontsize=10)
plt.gcf().set_figwidth(val=0.49 * 15.3978 * cm)
ax = plt.gca()
im = ax.imshow(g1, interpolation='none', extent=[-pm,pm,-pm,pm])
ax.set_aspect(1)
plt.scatter(e_1_xy[0],e_1_xy[1], c='blue', marker='x', linewidths=0.5, s=10)
plt.scatter(e_2_xy[0],e_2_xy[1], c='blue', marker='x', linewidths=0.5, s=10)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=8)
cbar.set_label(r"$g_{2}^{\left(1\right)}$", fontsize=10, rotation=-90, labelpad=15)
ax.set_xlabel(r"$x/\sigma$", fontsize=10, labelpad=1)
ax.set_ylabel(r"$y/\sigma$", fontsize=10, labelpad=-3)
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(path,'g1.png'), dpi=500, bbox_inches='tight')
plt.close()