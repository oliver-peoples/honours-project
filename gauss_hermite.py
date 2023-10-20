from scipy.special import hermite as physicistsHermite
import numpy as np
from parula import parula
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

cm = 1/2.54

matplotlib.rcParams['text.usetex'] = True

grid_x = 1750
grid_y = 1750
waists = 3
sqrt_2 = np.sqrt(2)

m = 2
n = 3
waist = 1.0
w_0 = 1.
I_0 = 1.

def rho(r, w) -> float:
    
    return 2 * r**2 / w**2

x_meshgrid, y_meshgrid = np.meshgrid(
    np.linspace(-waists * waist, waists * waist, grid_x),
    np.linspace(waists * waist, -waists * waist, grid_y)
)

h_m = physicistsHermite(m)
h_n = physicistsHermite(n)

gh_tem = I_0 * ((w_0 / waist)**2)
gh_tem *= ((h_m(x_meshgrid * sqrt_2 / waist) * np.exp(-x_meshgrid**2 / waist**2))**2)
gh_tem *= ((h_n(y_meshgrid * sqrt_2 / waist) * np.exp(-y_meshgrid**2 / waist**2))**2)

gh_tem *= 1. / np.max(gh_tem)

plt.figure()
plt.title(r'$\mathrm{TEM}_{mn},\;m=' + str(m) + r',\;n=' + str(n) + r'$', fontsize=10, pad=10)
plt.gcf().set_figwidth(val=0.49 * 15.3978 * cm)
ax = plt.gca()
im = ax.imshow(gh_tem, interpolation='none', extent=[-waists,waists,-waists,waists])
ax.set_aspect(1)
# plt.pcolormesh(x_meshgrid / waist, y_meshgrid / waist, gh_tem, cmap=parula)
plt.xlabel(r"$x/w$", fontsize=10)
plt.ylabel(r"$y/w$", fontsize=10)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label(r"$I_{mn}\left(xy\right)/I_{max}$", fontsize=10, rotation=-90, labelpad=15)
ax.set_xlabel(r"$x/w\left(0\right)$", fontsize=10, labelpad=1)
ax.set_ylabel(r"$y/w\left(0\right)$", fontsize=10, labelpad=-3)
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.gcf().set_figwidth(val=0.49 * 15.3978 * cm)
plt.savefig(f'gh-modes/m_{m}_n_{n}_i_mn.png', dpi=400, bbox_inches='tight')
plt.close()

# plt.figure()
# plt.title(r'$r=' + f'{r:.3f}' + r'\sigma$', fontsize=10)
# plt.gcf().set_figwidth(val=0.49 * 15.3978 * cm)
# ax = plt.gca()
# im = ax.imshow(g1, interpolation='none', extent=[-pm,pm,-pm,pm])
# ax.set_aspect(1)
# plt.scatter(e_1_xy[0],e_1_xy[1], c='blue', marker='x', linewidths=0.5, s=10)
# plt.scatter(e_2_xy[0],e_2_xy[1], c='blue', marker='x', linewidths=0.5, s=10)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# cbar = plt.colorbar(im, cax=cax)
# cbar.ax.tick_params(labelsize=8)
# cbar.set_label(r"$g_{2}^{\left(1\right)}$", fontsize=10, rotation=-90, labelpad=15)
# ax.set_xlabel(r"$x/\sigma$", fontsize=10, labelpad=1)
# ax.set_ylabel(r"$y/\sigma$", fontsize=10, labelpad=-3)
# ax.tick_params(labelsize=8)
# plt.tight_layout()
# plt.savefig(os.path.join(path,'g1.png'), dpi=500, bbox_inches='tight')
# plt.close()