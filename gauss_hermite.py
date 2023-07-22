from scipy.special import hermite as physicistsHermite
import numpy as np
from parula import parula
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['text.usetex'] = True

grid_x = 1750
grid_y = 1750
waists = 3
sqrt_2 = np.sqrt(2)

m = 1
n = 3
waist = 1.
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

plt.title(r'$\mathrm{TEM}_{mn},\;m=' + str(m) + r',\;n=' + str(n) + r'$', fontsize=28, pad=10)
plt.pcolormesh(x_meshgrid / waist, y_meshgrid / waist, gh_tem, cmap=parula)
plt.xlabel(r"$x/w$", fontsize=24)
plt.ylabel(r"$y/w$", fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
cbar = plt.colorbar(pad=0.01)
cbar.ax.tick_params(labelsize=18)
cbar.set_label(r"$I_{mn}\left(x,y\right)/I_{max}$", fontsize=24, rotation=-90, labelpad=28)
# cbar.set_ticks
plt.gca().set_aspect(1)
plt.tight_layout()
plt.savefig(f'gh-modes/m_{m}_m_{n}_i_mn.png', dpi=400, bbox_inches='tight')
plt.close()