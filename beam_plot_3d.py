import numpy as np
from parula import parula
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import KDTree
from mayavi import mlab
mlab.options.offscreen = True
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

matplotlib.rcParams['text.usetex'] = True

from qclm import Emitter, GaussLaguerre, GaussHermite, Detector, Solver, QuadTree, Rect, Point, genHermite, mlab_imshowColor
from scipy.optimize import minimize
from scipy.optimize import fmin

import os

path = os.path.dirname(__file__)

PLOT_IMAGE_PLANE = True

E_0 = 1.

m = 1
n = 1

gh_m = genHermite(m)
gh_n = genHermite(n)

wavelength = 632.8

w_0 = 400.0
n_s = 1.

x_slice = 0. 

wavelength *= 1e-9
wavenumber = 2 * np.pi / wavelength
w_0 *= 1e-9

z_r = np.pi * w_0**2 * n_s / wavelength

pm_z = z_r * 3

pm_w_0 = w_0 * 2.5

x_samples = 500
y_samples = 500
z_samples = 400

cam_az = 25
cam_el = 30

off_nadir_angles_deg = [35]
azimuth_angles_deg = [55]
roll_angles_deg = np.linspace(0,90,10)

z_offset = 0.9 * w_0

XYZ = True
XYZ_PRIME = True

OPTICAL_AXIS = False
BEAM_AXIS = False

#======================================================================================================================================================================================================
# Plotting, no user set variables below here
#======================================================================================================================================================================================================

def main() -> None:
    
    idx = 0

    for off_nadir_angle_deg in off_nadir_angles_deg:

        for azimuth_angle_deg in azimuth_angles_deg:
            
            for roll_angle_deg in roll_angles_deg:
                
                plot(off_nadir_angle_deg, azimuth_angle_deg, roll_angle_deg, idx)
                
                idx += 1
                
def plot(off_nadir_angle_deg, azimuth_angle_deg, roll_angle_deg, idx):
            
    x_linspace = np.linspace(-pm_w_0,pm_w_0,x_samples)
    y_linspace = np.linspace(-pm_w_0,pm_w_0,y_samples)
    z_linspace = np.linspace(-pm_z,pm_z,z_samples)

    normal = np.array([0.,0.,1.], dtype=np.float64).T

    img_space_y_meshgrid, img_space_x_meshgrid, img_space_z_meshgrid = np.meshgrid(x_linspace, y_linspace, z_linspace)

    y_rotation = Rotation.from_rotvec(off_nadir_angle_deg * np.pi / 180 * np.array([0,1,0], dtype=np.float64))
    z_rotation = Rotation.from_rotvec(azimuth_angle_deg * np.pi / 180 * np.array([0,0,1], dtype=np.float64))

    rotation = z_rotation.as_matrix() @ y_rotation.as_matrix()

    normal = rotation @ normal

    roll_rotation = Rotation.from_rotvec(roll_angle_deg * np.pi / 180 * normal).as_matrix()

    rotation = roll_rotation @ rotation

    x_basis = np.array([ 1.,0.,0. ], dtype=np.float64)
    y_basis = np.array([ 0.,1.,0. ], dtype=np.float64)
    z_basis = np.array([ 0.,0.,1. ], dtype=np.float64)

    beam_space_x_basis = rotation[:,0]
    beam_space_y_basis = rotation[:,1]
    beam_space_z_basis = rotation[:,2]

    beam_space_x_meshgrid = img_space_x_meshgrid * beam_space_x_basis[0] + img_space_y_meshgrid * beam_space_x_basis[1] + img_space_z_meshgrid * beam_space_x_basis[2]
    beam_space_y_meshgrid = img_space_x_meshgrid * beam_space_y_basis[0] + img_space_y_meshgrid * beam_space_y_basis[1] + img_space_z_meshgrid * beam_space_y_basis[2]
    beam_space_z_meshgrid = img_space_x_meshgrid * beam_space_z_basis[0] + img_space_y_meshgrid * beam_space_z_basis[1] + img_space_z_meshgrid * beam_space_z_basis[2]

    beam_space_z_meshgrid -= z_offset
                
    w_z_meshgrid = w_0 * np.sqrt(1 + (beam_space_z_meshgrid / z_r)**2)

    h_m = gh_m(np.sqrt(2) * beam_space_x_meshgrid / w_z_meshgrid)
    h_n = gh_n(np.sqrt(2) * beam_space_y_meshgrid / w_z_meshgrid)

    scalar_comp = E_0 * w_0 / w_z_meshgrid * h_m * h_n

    sum_xy_squares = beam_space_x_meshgrid**2 + beam_space_y_meshgrid**2

    inv_w_squared = 1 / w_z_meshgrid**2

    inv_r = beam_space_z_meshgrid / (beam_space_z_meshgrid**2 + z_r**2)

    vergence_comp = 1.j * wavenumber * inv_r / 2

    wavenumber_comp = 1.j * wavenumber * beam_space_z_meshgrid

    gouy_phase_shift = 1.j * (n + m + 1) * np.arctan(beam_space_z_meshgrid / z_r)

    exp_component = np.exp(-sum_xy_squares * (inv_w_squared + vergence_comp) - wavenumber_comp - gouy_phase_shift)

    intensity = np.abs(scalar_comp * exp_component)**2

    max_intensity = np.max(intensity)
    intensity /=max_intensity

    fig = mlab.figure(size=(2000,2000), bgcolor=(1,1,1))

    src = mlab.pipeline.scalar_field(img_space_x_meshgrid * 1e9, img_space_y_meshgrid * 1e9, img_space_z_meshgrid * 1e9, intensity)
    mlab.pipeline.iso_surface(src, contours=[0.75], opacity=0.75, color=parula(0.75)[:3])
    mlab.pipeline.iso_surface(src, contours=[0.5], opacity=0.5, color=parula(0.5)[:3])
    mlab.pipeline.iso_surface(src, contours=[0.15], opacity=0.25, color=parula(0.15)[:3])

    cam_az_rad = np.pi * cam_az / 180
    cam_el_rad = np.pi * cam_el / 180

    cam = mlab.gcf().scene.camera
    pos = cam.position
    scale = 1
    cam_norm = scale * np.linalg.norm(cam.position)

    print(pos)

    cam.position = np.array([
        cam_norm * np.cos(cam_az_rad) * np.cos(cam_el_rad),
        cam_norm * np.sin(cam_az_rad) * np.cos(cam_el_rad),
        cam_norm * np.sin(cam_el_rad)
    ])

    # 2d slice

    if PLOT_IMAGE_PLANE:

        img_space_y_meshgrid, img_space_x_meshgrid = np.meshgrid(
            x_linspace,
            y_linspace,
        )

        img_space_z_meshgrid = np.zeros_like(img_space_x_meshgrid)

        beam_space_x_meshgrid = img_space_x_meshgrid * beam_space_x_basis[0] + img_space_y_meshgrid * beam_space_x_basis[1] + img_space_z_meshgrid * beam_space_x_basis[2]
        beam_space_y_meshgrid = img_space_x_meshgrid * beam_space_y_basis[0] + img_space_y_meshgrid * beam_space_y_basis[1] + img_space_z_meshgrid * beam_space_y_basis[2]
        beam_space_z_meshgrid = img_space_x_meshgrid * beam_space_z_basis[0] + img_space_y_meshgrid * beam_space_z_basis[1] + img_space_z_meshgrid * beam_space_z_basis[2]

        beam_space_z_meshgrid -= z_offset

        w_z_meshgrid = w_0 * np.sqrt(1 + (beam_space_z_meshgrid / z_r)**2)

        h_m = gh_m(np.sqrt(2) * beam_space_x_meshgrid / w_z_meshgrid)
        h_n = gh_n(np.sqrt(2) * beam_space_y_meshgrid / w_z_meshgrid)

        scalar_comp = E_0 * w_0 / w_z_meshgrid * h_m * h_n

        sum_xy_squares = beam_space_x_meshgrid**2 + beam_space_y_meshgrid**2

        inv_w_squared = 1 / w_z_meshgrid**2

        inv_r = beam_space_z_meshgrid / (beam_space_z_meshgrid**2 + z_r**2)

        vergence_comp = 1.j * wavenumber * inv_r / 2

        wavenumber_comp = 1.j * wavenumber * beam_space_z_meshgrid

        gouy_phase_shift = 1.j * (n + m + 1) * np.arctan(beam_space_z_meshgrid / z_r)

        exp_component = np.exp(-sum_xy_squares * (inv_w_squared + vergence_comp) - wavenumber_comp - gouy_phase_shift)

        intensity = np.abs(scalar_comp * exp_component)**2

        intensity /= max_intensity

        s = parula(intensity)

        s *= 255
        
        mlab_imshowColor(s[:, :, :3], 255, extent=[-pm_w_0 * 1e9,pm_w_0 * 1e9,-pm_w_0 * 1e9,pm_w_0 * 1e9,0,0])
        
        plt.title(r'$\mathrm{TEM}_{mn},\;m=' + str(m) + r',\;n=' + str(n) + r'$', fontsize=28, pad=10)
        plt.pcolormesh(img_space_x_meshgrid / w_0, img_space_y_meshgrid / w_0, intensity, cmap=parula)
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
        plt.savefig(os.path.join(path,'animation-frames',f'image_plane_{m}_{n}_{idx:04d}.png'), dpi=400, bbox_inches='tight')
        plt.close()

    if XYZ:
            
        x_axis = np.array([
            [ 0,0,0 ],
            [*(x_basis * 1.5 * pm_w_0 * 1e9)]
        ])

        mlab.plot3d(
            x_axis[:,0],
            x_axis[:,1],
            x_axis[:,2],
            color=(1.0,0.0,0.0),
            tube_radius=10,
            tube_sides=30
        )

        y_axis = np.array([
            [ 0,0,0 ],
            [*(y_basis * 1.5 * pm_w_0 * 1e9)]
        ])

        mlab.plot3d(
            y_axis[:,0],
            y_axis[:,1],
            y_axis[:,2],
            color=(0.0,1.0,0.0),
            tube_radius=10,
            tube_sides=30
        )

        z_axis = np.array([
            [ 0,0,0 ],
            [*(z_basis * 1.5 * pm_w_0 * 1e9)]
        ])

        mlab.plot3d(
            z_axis[:,0],
            z_axis[:,1],
            z_axis[:,2],
            color=(0.0,0.0,1.0),
            tube_radius=10,
            tube_sides=30
        )
        
    if XYZ_PRIME:

        beam_space_x_axis = np.array([
            [ 0,0,0 ],
            [*(beam_space_x_basis * 1.5 * pm_w_0 * 1e9)]
        ])

        mlab.plot3d(
            beam_space_x_axis[:,0],
            beam_space_x_axis[:,1],
            beam_space_x_axis[:,2],
            color=(1.0,0.0,1.0),
            tube_radius=10,
            tube_sides=30
        )

        beam_space_y_axis = np.array([
            [ 0,0,0 ],
            [*(beam_space_y_basis * 1.5 * pm_w_0 * 1e9)]
        ])

        mlab.plot3d(
            beam_space_y_axis[:,0],
            beam_space_y_axis[:,1],
            beam_space_y_axis[:,2],
            color=(1.0,1.0,0.0),
            tube_radius=10,
            tube_sides=30
        )

        beam_space_z_axis = np.array([
            [ 0,0,0 ],
            [*(beam_space_z_basis * 1.5 * pm_w_0 * 1e9)]
        ])

        mlab.plot3d(
            beam_space_z_axis[:,0],
            beam_space_z_axis[:,1],
            beam_space_z_axis[:,2],
            color=(0.0,1.0,1.0),
            tube_radius=10,
            tube_sides=30
        )    
        
    if not XYZ and OPTICAL_AXIS:
        
        z_axis = np.array([
            [*(-z_basis * 1. * pm_z * 1e9)],
            [*(z_basis * 1. * pm_z * 1e9)]
        ])

        mlab.plot3d(
            z_axis[:,0],
            z_axis[:,1],
            z_axis[:,2],
            color=(0.0,1.0,1.0),
            tube_radius=10,
            tube_sides=30
        )
        
    if not XYZ and BEAM_AXIS:
        
        beam_space_z_axis = np.array([
            [*(-beam_space_z_basis * 1. * pm_z * 1e9) ],
            [*(beam_space_z_basis * 1. * pm_z * 1e9)]
        ])

        mlab.plot3d(
            beam_space_z_axis[:,0],
            beam_space_z_axis[:,1],
            beam_space_z_axis[:,2],
            color=(0.0,0.0,1.0),
            tube_radius=10,
            tube_sides=30
        )  

    f = mlab.gcf()
    f.scene._lift()
    # img_array = mlab.screenshot(figure=f, mode='rgba')
    mlab.savefig(os.path.join(path,'animation-frames',f'beam_3d_{m}_{n}_{idx:04d}.png'))
    
if __name__ == '__main__':
    
    main()
    
# mlab.show()