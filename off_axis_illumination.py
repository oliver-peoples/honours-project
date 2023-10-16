import numpy as np
from parula import parula
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

matplotlib.rcParams['text.usetex'] = True

from qclm import Emitter, GaussLaguerre, GaussHermite, Detector, Solver, QuadTree, Rect, Point, genHermite
from scipy.optimize import minimize
from scipy.optimize import fmin

import os

path = os.path.dirname(__file__)

E_0 = 1.

m = 0
n = 1

wavelength = 632.8

w_0 = 200.0
n_s = 1.

off_nadir_angle_deg = 45
azimuth_deg = 0
roll_deg = 0

# change normal

off_nadir_angle_rad = off_nadir_angle_deg * np.pi / 180
azimuth_rad = azimuth_deg * np.pi / 180
roll_rad = roll_deg * np.pi / 180

tem_frame = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
    ], dtype=np.float64)

off_nadir_trans = Rotation.from_rotvec(off_nadir_angle_rad * np.array([0,1,0]))

tem_frame = off_nadir_trans.as_matrix() @ tem_frame

azimuth_trans = Rotation.from_rotvec(azimuth_rad * np.array([0,0,1]))

tem_frame = azimuth_trans.as_matrix() @ tem_frame

roll_trans = Rotation.from_rotvec(roll_rad * np.array(tem_frame[:,2]))

tem_frame = roll_trans.as_matrix() @ tem_frame

gh_m = genHermite(m)
gh_n = genHermite(n)

x_slice = 0. 

wavelength *= 1e-9
wavenumber = 2 * np.pi / wavelength
w_0 *= 1e-9

z_r = np.pi * w_0**2 * n_s / wavelength

pm_w_0 = w_0 * 2

samples = 500

# slice in the image plane

x_linspace = np.linspace(-pm_w_0, pm_w_0, samples)
y_linspace = np.linspace(-pm_w_0, pm_w_0, samples)

x_meshgrid, y_meshgrid = np.meshgrid(x_linspace, y_linspace)
z_meshgrid = np.zeros_like(x_meshgrid)

stack = np.array([x_meshgrid,y_meshgrid,z_meshgrid])

stack_test = np.tensordot(tem_frame,stack,axes=(1))

for row in range(np.shape(x_meshgrid)[0]):
    
    for col in range(np.shape(x_meshgrid)[1]):
        
        stack[:,row,col] = tem_frame @ stack[:,row,col]
        
print(np.max(stack - stack_test))

x_meshgrid = stack[0,:,:]
y_meshgrid = stack[1,:,:]
z_meshgrid = stack[2,:,:]

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

plt.pcolormesh(x_meshgrid / w_0, y_meshgrid / w_0, intensity, cmap=parula)
plt.xlabel(r"$x/w_{0}$", fontsize=18)
plt.ylabel(r"$y/w_{0}$", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
cbar = plt.colorbar(pad=0.01)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(r"$I_{mn}\left(x,y\right)/I_{max}$", fontsize=18, rotation=-90, labelpad=28)
plt.gca().set_aspect(1)
plt.tight_layout()
plt.savefig(f'gh-modes/off_axis_m_{m}_m_{n}_i_mn.png', dpi=400, bbox_inches='tight')
plt.close()
