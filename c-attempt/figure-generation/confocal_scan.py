import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from parula import parula
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

cm = 1/2.54

d_w = 1.

waists = 1.5

path = os.path.dirname(__file__)

matplotlib.rcParams['text.usetex'] = True

emitter_xy = np.array([
    [-0.6300,-0.1276],
    [0.5146,-0.5573]
], dtype=np.float64)

emitter_brightness = [ 1.0,0.3617 ]

#=========================================================================

e_1_xy = emitter_xy[0,:]
e_2_xy = emitter_xy[1,:]

P_1 = emitter_brightness[0]
P_2 = emitter_brightness[1]

x_linspace = np.linspace(-waists * d_w,waists * d_w, 1000)
y_linspace = np.linspace(waists * d_w,-waists * d_w, 1000)

x_meshgrid, y_meshgrid = np.meshgrid(x_linspace, y_linspace)

e_1_r_meshgrid = np.sqrt((x_meshgrid - e_1_xy[0])**2 + (y_meshgrid - e_1_xy[1])**2)
e_2_r_meshgrid = np.sqrt((x_meshgrid - e_2_xy[0])**2 + (y_meshgrid - e_2_xy[1])**2)

p_1_meshgrid = P_1 * np.exp(-(e_1_r_meshgrid**2/2)/(2*d_w**2))
p_2_meshgrid = P_2 * np.exp(-(e_2_r_meshgrid**2/2)/(2*d_w**2))

g1 = (p_1_meshgrid + p_2_meshgrid) / (P_1 + P_2)

alpha = p_1_meshgrid / p_2_meshgrid

g2 = (2 * alpha) / (1 + alpha)**2

plt.figure()

plt.gcf().set_figwidth(val=0.99 * 15.3978 * cm)

plt.subplot(1,2,1)

ax = plt.gca()
im = ax.imshow(g1, interpolation='none', extent=[-waists,waists,-waists,waists])
ax.set_aspect(1)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=8)
cbar.set_label(r"$g_{2}^{\left(1\right)}$", fontsize=10, rotation=-90, labelpad=15)
ax.set_xlabel(r"$x/\sigma$", fontsize=10, labelpad=1)
ax.set_ylabel(r"$y/\sigma$", fontsize=10, labelpad=-3)
ax.tick_params(labelsize=8)
plt.scatter(e_1_xy[0],e_1_xy[1], c='blue', marker='x', linewidths=0.5, s=10)
plt.scatter(e_2_xy[0],e_2_xy[1], c='blue', marker='x', linewidths=0.5, s=10)
plt.tight_layout()

plt.subplot(1,2,2)

plt.yticks()
ax = plt.gca()
im = ax.imshow(g2, interpolation='none', extent=[-waists,waists,-waists,waists])
ax.set_aspect(1)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
ax.contour(x_meshgrid, y_meshgrid, g2, linewidths=1, colors='k', levels=[0.2,0.25,0.3,0.35,0.4,0.45])
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=8)
cbar.set_label(r"$g_{2}^{\left(2\right)}$", fontsize=10, rotation=-90, labelpad=15)
ax.set_xlabel(r"$x/\sigma$", fontsize=10, labelpad=1)
ax.yaxis.set_major_locator(ticker.NullLocator())
plt.scatter(e_1_xy[0],e_1_xy[1], c='blue', marker='x', linewidths=0.5, s=10)
plt.scatter(e_2_xy[0],e_2_xy[1], c='blue', marker='x', linewidths=0.5, s=10)
# ax.yaxis.set_major_locator(ticker.NullLocator())
# ax.set_ylabel(r"$y/\sigma$", fontsize=10, labelpad=-3)
ax.tick_params(labelsize=8)
plt.tight_layout()

plt.subplots_adjust(wspace=0.25)
plt.savefig(os.path.join(path,'g1_g2_scan.png'), dpi=500, bbox_inches='tight')
plt.close()

# P02 = 0.3617;
# x1 = [-0.6300,   -0.1276];
# x2 = [0.5146,   -0.5573];