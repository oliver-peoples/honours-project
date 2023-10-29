import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import KDTree
# from mayavi import mlab
# mlab.options.offscreen = True
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
matplotlib.rcParams['text.usetex'] = True

import os

path = os.path.dirname(__file__)

import sys

sys.path.append(os.path.abspath(os.path.join(path,os.path.pardir)))

from parula import parula

x_basis = np.array([ 1.,0.,0. ], dtype=np.float64)
y_basis = np.array([ 0.,1.,0. ], dtype=np.float64)
z_basis = np.array([ 0.,0.,1. ], dtype=np.float64)

def main() -> None:
    
    meta = np.genfromtxt(os.path.join(path,'meta.out'), delimiter=',')
    
    emitter_xy = np.genfromtxt(os.path.join(path,'emitter_xy.csv'), delimiter=',', skip_header=1)
    
    # print(np.linalg.norm(emitter_xy[0,:] - emitter_xy[1,:]))

    beam_centers = np.genfromtxt(os.path.join(path,'beam_centers.csv'), delimiter=',', skip_header=1)
    
    num_x, num_y, num_z = meta[0,:]
    
    num_x = int(num_x)
    num_y = int(num_y)
    num_z = int(num_z)
    
    bl_corner = meta[1,:]
    tr_corner = meta[2,:]
    
    x_linspace = np.linspace(bl_corner[0],tr_corner[0], num_x)
    y_linspace = np.linspace(bl_corner[1],tr_corner[1], num_y)
    z_linspace = np.linspace(bl_corner[2],tr_corner[2], num_z)

    normal = np.array([0.,0.,1.], dtype=np.float64).T

    if num_z > 1:
        
        img_space_y_meshgrid, img_space_x_meshgrid, img_space_z_meshgrid = np.meshgrid(y_linspace, x_linspace, z_linspace)
        
        # intensity_field = np.fromfile(os.path.join(path,'tem_intensities.bin'), np.float64, count=num_x * num_y * num_z)
        
        # intensity_field = intensity_field.reshape((num_x,num_y,num_z))
        
        # max_intensity = np.max(intensity_field)
        
        # intensity_field /= max_intensity

        # fig = mlab.figure(size=(2000,2000), bgcolor=(1,1,1))

        # src = mlab.pipeline.scalar_field(img_space_y_meshgrid, img_space_x_meshgrid, img_space_z_meshgrid, intensity_field)
        # mlab.pipeline.iso_surface(src, contours=[0.75], opacity=1, color=parula(0.75)[:3])
        # mlab.pipeline.iso_surface(src, contours=[0.5], opacity=0.5, color=parula(0.5)[:3])
        # # mlab.pipeline.iso_surface(src, contours=[0.15], opacity=0.25, color=parula(0.15)[:3])

        # cam_az = 45
        # cam_el = 30

        # cam_az_rad = np.pi * cam_az / 180
        # cam_el_rad = np.pi * cam_el / 180

        # cam = mlab.gcf().scene.camera
        # pos = cam.position
        # scale = 0.75
        # cam_norm = scale * np.linalg.norm(cam.position)

        # print(pos)

        # cam.position = np.array([
        #     cam_norm * np.cos(cam_az_rad) * np.cos(cam_el_rad),
        #     cam_norm * np.sin(cam_az_rad) * np.cos(cam_el_rad),
        #     cam_norm * np.sin(cam_el_rad)
        # ])

        # x_axis = np.array([
        #     [ 0,0,0 ],
        #     [*(x_basis * 0.1)]
        # ])

        # mlab.plot3d(
        #     x_axis[:,0],
        #     x_axis[:,1],
        #     x_axis[:,2],
        #     color=(1.0,0.0,0.0),
        #     tube_radius=0.001,
        #     tube_sides=30
        # )

        # y_axis = np.array([
        #     [ 0,0,0 ],
        #     [*(y_basis * 0.1)]
        # ])

        # mlab.plot3d(
        #     y_axis[:,0],
        #     y_axis[:,1],
        #     y_axis[:,2],
        #     color=(0.0,1.0,0.0),
        #     tube_radius=0.001,
        #     tube_sides=30
        # )

        # z_axis = np.array([
        #     [ 0,0,0 ],
        #     [*(z_basis * 0.1)]
        # ])
        
        # mlab.plot3d(
        #     z_axis[:,0],
        #     z_axis[:,1],
        #     z_axis[:,2],
        #     color=(0.0,1.0,1.0),
        #     tube_radius=0.001,
        #     tube_sides=30
        # )
        
        # cam_az = 45
        # cam_el = 30

        # cam_az_rad = np.pi * cam_az / 180
        # cam_el_rad = np.pi * cam_el / 180

        # cam = mlab.gcf().scene.camera
        # pos = cam.position
        # scale = 0.75
        # cam_norm = scale * np.linalg.norm(cam.position)

        # print(pos)

        # cam.position = np.array([
        #     cam_norm * np.cos(cam_az_rad) * np.cos(cam_el_rad),
        #     cam_norm * np.sin(cam_az_rad) * np.cos(cam_el_rad),
        #     cam_norm * np.sin(cam_el_rad)
        # ])
        
        # mlab.show()
        
    else:
        
        
        i_fn = np.fromfile(os.path.join(path,'chi_2.bin'), dtype=np.float64).reshape((num_y,num_x))

        # print(np.max(i_fn))
        # i_fn /= np.max(i_fn)

        x_linspace = np.linspace(bl_corner[0],tr_corner[0],num_x)
        y_linspace = np.linspace(bl_corner[1],tr_corner[1],num_y)

        x_meshgrid, y_meshgrid = np.meshgrid(x_linspace,y_linspace)

        # plt.title(r'$\mathrm{TEM}_{mn},\;m=' + str(mn[0]) + r',\;n=' + str(mn[1]) + r'$', fontsize=28, pad=10)
        
        # plt.scatter(emitter_xy[:,1],emitter_xy[:,2], c='magenta', marker='.', s=1)
        
        # plt.scatter(x1s[:,1],x1s[:,2], c='cyan', marker='.', s=1)
        plt.scatter(beam_centers[:,1],beam_centers[:,2], c='k', marker='+', s=10, linewidths=0.5)
    
        plt.pcolormesh(x_meshgrid, y_meshgrid, i_fn, cmap=parula)
        plt.xlabel(r"$\mathrm{Hypothesis\;} x_{2}\;[\sigma]$", fontsize=20)
        plt.ylabel(r"$\mathrm{Hypothesis\;} y_{2}\;[\sigma]$", fontsize=20)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        cbar = plt.colorbar(pad=0.01)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(r"$\chi^{2}$", fontsize=20, rotation=-90, labelpad=20)
        plt.scatter(emitter_xy[1,1], emitter_xy[1,2], c='red', marker='+', linewidths=1.5, s=40)
        # cbar.set_ticks
        plt.gca().set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(path,'chi_2.png'), dpi=200, bbox_inches='tight')
        plt.close()
        
if __name__ == '__main__':
    
    main()