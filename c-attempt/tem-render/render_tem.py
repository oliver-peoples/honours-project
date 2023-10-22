from scipy.special import hermite as physicistsHermite
import numpy as np
from parula import parula
import matplotlib.pyplot as plt
import matplotlib
import os

path = os.path.dirname(__file__)

matplotlib.rcParams['text.usetex'] = True

num_xy = np.fromfile(os.path.join(path,'i_fn_out.bin'), dtype=np.int32, count=2)
pm_xy = np.fromfile(os.path.join(path,'i_fn_out.bin'), dtype=np.float64, count=2, offset=8)
mn = np.fromfile(os.path.join(path,'i_fn_out.bin'), dtype=np.int32, count=2, offset=24)

i_fn = np.fromfile(os.path.join(path,'i_fn_out.bin'), dtype=np.float64, offset=32).reshape((num_xy[1],num_xy[0]))

print(np.max(i_fn))
i_fn /= np.max(i_fn)


x_linspace = np.linspace(-pm_xy[0],pm_xy[0],num_xy[0])
y_linspace = np.linspace(pm_xy[1],-pm_xy[1],num_xy[1])

x_meshgrid, y_meshgrid = np.meshgrid(x_linspace,y_linspace)

plt.title(r'$\mathrm{TEM}_{mn},\;m=' + str(mn[0]) + r',\;n=' + str(mn[1]) + r'$', fontsize=28, pad=10)
plt.pcolormesh(x_meshgrid, y_meshgrid, i_fn, cmap=parula)
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
plt.savefig(f'renders/m_{mn[0]}_m_{mn[1]}_i_mn.png', dpi=400, bbox_inches='tight')
plt.close()

# print(num_xy, pm_xy)