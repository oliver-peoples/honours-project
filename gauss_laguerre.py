from scipy.special import genlaguerre as genLaguerre
import numpy as np
from parula import parula
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['text.usetex'] = True

grid_x = 1750
grid_y = 1750
waists = 3

order = 3
index = 1
waist = 1.
I_0 = 1.

def rho(r, w) -> float:
    
    return 2 * r**2 / w**2

x_meshgrid, y_meshgrid = np.meshgrid(
    np.linspace(-waists * waist, waists * waist, grid_x),
    np.linspace(waists * waist, -waists * waist, grid_y)
)

r_meshgrid = np.sqrt(x_meshgrid**2 + y_meshgrid**2)
phi_meshgrid = np.arctan2(y_meshgrid, x_meshgrid)

# plt.title(r'$\mathrm{TEM}_{pl},\;p=' + str(order) + r',\;l=' + str(index) + r'$', fontsize=28)
# plt.pcolormesh(x_meshgrid / waist, y_meshgrid / waist, r_meshgrid)
# plt.xlabel(r"$x/w$", fontsize=24)
# plt.ylabel(r"$y/w$", fontsize=24)
# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)
# cbar = plt.colorbar(pad=0.01)
# cbar.ax.tick_params(labelsize=18)
# cbar.set_label(r"$r$", fontsize=24, rotation=-90, labelpad=28)
# # cbar.set_ticks
# plt.gca().set_aspect(1)
# plt.tight_layout()
# plt.savefig(f'gl-modes/p_{order}_l_{index}_r.png', dpi=600, bbox_inches='tight')
# plt.close()

# plt.title(r'$\mathrm{TEM}_{pl},\;p=' + str(order) + r',\;l=' + str(index) + r'$', fontsize=28)
# plt.pcolormesh(x_meshgrid / waist, y_meshgrid / waist, phi_meshgrid)
# plt.xlabel(r"$x/w$", fontsize=24)
# plt.ylabel(r"$y/w$", fontsize=24)
# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)
# cbar = plt.colorbar(pad=0.01)
# cbar.ax.tick_params(labelsize=18)
# cbar.set_label(r"$\phi$", fontsize=24, rotation=-90, labelpad=28)
# # cbar.set_ticks
# plt.gca().set_aspect(1)
# plt.tight_layout()
# plt.savefig(f'gl-modes/p_{order}_l_{index}_phi.png', dpi=600, bbox_inches='tight')
# plt.close()

rho_meshgrid = rho(r_meshgrid, waist)
gen_laguerre = genLaguerre(order, index)
gl_tem = I_0 * (rho_meshgrid**index) * (gen_laguerre(rho_meshgrid)**2) * (np.cos(index * phi_meshgrid)**2) * np.exp(-rho_meshgrid)
gl_tem /= np.max(gl_tem)

plt.title(r'$\mathrm{TEM}_{pl},\;p=' + str(order) + r',\;l=' + str(index) + r'$', fontsize=28, pad=10)
plt.pcolormesh(x_meshgrid / waist, y_meshgrid / waist, gl_tem, cmap=parula)
plt.xlabel(r"$x/w$", fontsize=24)
plt.ylabel(r"$y/w$", fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
cbar = plt.colorbar(pad=0.01)
cbar.ax.tick_params(labelsize=18)
cbar.set_label(r"$I_{pl}\left(\rho,\phi\right)/I_{max}$", fontsize=24, rotation=-90, labelpad=28)
# cbar.set_ticks
plt.gca().set_aspect(1)
plt.tight_layout()
plt.savefig(f'gl-modes/p_{order}_l_{index}_i_pl.png', dpi=400, bbox_inches='tight')
plt.close()