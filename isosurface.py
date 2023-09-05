import numpy as np
from parula import parula
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import KDTree
from mayavi import mlab
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['text.usetex'] = True

from qclm import Emitter, GaussLaguerre, GaussHermite, Detector, Solver, QuadTree, Rect, Point, genHermite, meshgrid2
from scipy.optimize import minimize
from scipy.optimize import fmin

import os

path = os.path.dirname(__file__)

E_0 = 1.

m = 0
n = 1

gh_m = genHermite(m)
gh_n = genHermite(n)

wavelength = 632.8

w_0 = 200.0
n_s = 1.

x_slice = 0. 

wavelength *= 1e-9
wavenumber = 2 * np.pi / wavelength
w_0 *= 1e-9

z_r = np.pi * w_0**2 * n_s / wavelength

pm_z = z_r * 3

pm_w_0 = w_0 * 1.5

x_samples = 500
y_samples = 500
z_samples = 1000

x_linspace = np.linspace(-pm_w_0, pm_w_0, x_samples)
y_linspace = np.linspace(-pm_w_0, pm_w_0, y_samples)
z_linspace = np.linspace(-pm_z, pm_z, z_samples)

y_meshgrid, x_meshgrid, z_meshgrid = np.meshgrid(x_linspace, y_linspace, z_linspace)

# y_meshgrid, x_meshgrid, z_meshgrid = np.mgrid[x_linspace, y_linspace, z_linspace]

print(np.shape(x_meshgrid))

# z_meshgrid, y_meshgrid = np.meshgrid(z_linspace, y_linspace)
# x_meshgrid = np.zeros_like(y_meshgrid) + x_slice

w_z_meshgrid = w_0 * np.sqrt(1 + (z_meshgrid / z_r)**2)

h_m = gh_m(np.sqrt(2) * x_meshgrid / w_z_meshgrid)
h_n = gh_n(np.sqrt(2) * y_meshgrid / w_z_meshgrid)

scalar_comp = E_0 * w_0 / w_z_meshgrid * h_m * h_n

sum_xy_squares = x_meshgrid**2 + y_meshgrid**2

inv_w_squared = 1 / w_z_meshgrid**2

inv_r = z_meshgrid / (z_meshgrid**2 + z_r**2)

vergence_comp = 1.j * wavenumber * inv_r / 2

wavenumber_comp = 1.j * wavenumber * z_meshgrid

gouy_phase_shift = 1.j * (n + m + 1) * np.arctan(z_meshgrid / z_r)

exp_component = np.exp(-sum_xy_squares * (inv_w_squared + vergence_comp) - wavenumber_comp - gouy_phase_shift)

intensity = np.abs(scalar_comp * exp_component)**2

intensity /= np.max(intensity)

src = mlab.pipeline.scalar_field(intensity)
mlab.pipeline.iso_surface(src, contours=[intensity.min()+0.1*intensity.ptp(), ], opacity=0.3)
mlab.pipeline.iso_surface(src, contours=[intensity.max()-0.1*intensity.ptp(), ],)

mlab.show()

# plt.pcolormesh(z_meshgrid / z_r, y_meshgrid / w_0, intensity, cmap=parula)
# plt.plot([1,0],[-2,2], 'k--')
# plt.xlabel(r"$z/z_{R}$", fontsize=18)
# plt.ylabel(r"$y/w_{0}$", fontsize=18)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# cbar = plt.colorbar(pad=0.01)
# cbar.ax.tick_params(labelsize=12)
# cbar.set_label(r"$I_{mn}\left(x,y\right)/I_{max}$", fontsize=18, rotation=-90, labelpad=28)
# plt.gca().set_aspect(1)
# plt.tight_layout()
# plt.savefig(f'gh-modes/skew_side_on_m_{m}_m_{n}_i_mn.png', dpi=400, bbox_inches='tight')
# plt.close()

# # flat slice

# x_linspace = np.linspace(-pm_w_0, pm_w_0, z_samples)
# y_linspace = np.linspace(-pm_w_0, pm_w_0, y_samples)

# x_meshgrid, y_meshgrid = np.meshgrid(x_linspace, y_linspace)
# z_meshgrid = np.zeros_like(x_meshgrid)

# w_z_meshgrid = w_0 * np.sqrt(1 + (z_meshgrid / z_r)**2)

# h_m = gh_m(np.sqrt(2) * x_meshgrid / w_z_meshgrid)
# h_n = gh_n(np.sqrt(2) * y_meshgrid / w_z_meshgrid)

# scalar_comp = E_0 * w_0 / w_z_meshgrid * h_m * h_n

# sum_xy_squares = x_meshgrid**2 + y_meshgrid**2

# inv_w_squared = 1 / w_z_meshgrid**2

# inv_r = z_meshgrid / (z_meshgrid**2 + z_r**2)

# vergence_comp = 1.j * wavenumber * inv_r / 2

# wavenumber_comp = 1.j * wavenumber * z_meshgrid

# gouy_phase_shift = 1.j * (n + m + 1) * np.arctan(z_meshgrid / z_r)

# exp_component = np.exp(-sum_xy_squares * (inv_w_squared + vergence_comp) - wavenumber_comp - gouy_phase_shift)

# intensity = np.abs(scalar_comp * exp_component)**2

# intensity /= np.max(intensity)

# plt.pcolormesh(x_meshgrid / w_0, y_meshgrid / w_0, intensity, cmap=parula)
# plt.plot([0,0],[-2,2], 'k--')
# plt.xlabel(r"$x/w_{0}$", fontsize=18)
# plt.ylabel(r"$y/w_{0}$", fontsize=18)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# cbar = plt.colorbar(pad=0.01)
# cbar.ax.tick_params(labelsize=12)
# cbar.set_label(r"$I_{mn}\left(x,y\right)/I_{max}$", fontsize=18, rotation=-90, labelpad=28)
# plt.gca().set_aspect(1)
# plt.tight_layout()
# plt.savefig(f'gh-modes/front_on_m_{m}_m_{n}_i_mn.png', dpi=400, bbox_inches='tight')
# plt.close()