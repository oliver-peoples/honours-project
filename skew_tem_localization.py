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

from qclm import Emitter, Detector, genHermite
from scipy.optimize import minimize
from scipy.optimize import fmin

import os

path = os.path.dirname(__file__)

PLOT_3D = False
PLOT_2D = False

XYZ = True
XYZ_PRIME = True

OPTICAL_AXIS = False
BEAM_AXIS = False

cam_az = 25
cam_el = 30

E_0 = 1.

wavelength = 632.8 # HeNe

w_0 = 400.0
n_s = 1.

wavelength *= 1e-9
wavenumber = 2 * np.pi / wavelength
w_0 *= 1e-9

z_r = np.pi * w_0**2 * n_s / wavelength

pm_z = z_r * 3

pm_w_0 = w_0 * 2.5

x_2d_samples = 500
y_2d_samples = 500

x_3d_samples = 250
y_3d_samples = 250
z_3d_samples = 500

def main():
    
    detector = Detector(np.array([0.,0.,0.]))
    
    detector.waist = 1.5 * w_0
    
    e_1 = Emitter(
        np.array([
            -0.75 * w_0,-0.25 * w_0,0.
        ], dtype=np.float64),
        1.
    )
    
    e_2 = Emitter(
        np.array([
            1. * w_0,1. * w_0,0.
        ], dtype=np.float64),
        0.5
    )
    
    modes = [
        {'name':'structure_0','mn':[0,0],'trans_deg':[35,0,0],'offset_z':(0.9 * w_0)},
        {'name':'structure_1','mn':[1,1],'trans_deg':[55,55,0],'offset_z':(0.9 * w_0)},
        {'name':'structure_2','mn':[1,1],'trans_deg':[55,55,10],'offset_z':(0.9 * w_0)},
        {'name':'structure_3','mn':[1,1],'trans_deg':[55,55,20],'offset_z':(0.9 * w_0)},
        {'name':'structure_4','mn':[1,1],'trans_deg':[55,55,30],'offset_z':(0.9 * w_0)},
        {'name':'structure_5','mn':[1,1],'trans_deg':[55,55,40],'offset_z':(0.9 * w_0)},
    ]
    
    #==================================================================================================================================================================================================
    # Initialize the modes
    #==================================================================================================================================================================================================

    for mode_idx in range(len(modes)):
        
        y_rotation = Rotation.from_rotvec(modes[mode_idx]['trans_deg'][0] * np.pi / 180 * np.array([0,1,0], dtype=np.float64))
        z_rotation = Rotation.from_rotvec(modes[mode_idx]['trans_deg'][1] * np.pi / 180 * np.array([0,0,1], dtype=np.float64))

        rotation = z_rotation.as_matrix() @ y_rotation.as_matrix()
        
        normal = np.array([0.,0.,1.], dtype=np.float64).T

        normal = rotation @ normal

        roll_rotation = Rotation.from_rotvec(modes[mode_idx]['trans_deg'][2] * np.pi / 180 * normal).as_matrix()

        rotation = roll_rotation @ rotation

        modes[mode_idx]['so3transform'] = rotation
        
        x_linspace = np.linspace(-pm_w_0,pm_w_0,x_2d_samples)
        y_linspace = np.linspace(-pm_w_0,pm_w_0,y_2d_samples)
        
        modes[mode_idx]['gh_m'] = genHermite(modes[mode_idx]['mn'][0])
        modes[mode_idx]['gh_n'] = genHermite(modes[mode_idx]['mn'][1])
        
    #======================================================================================================================================================================================================
    # Get truth G1 G2 values
    #======================================================================================================================================================================================================

    g1_truth = np.ndarray((len(modes),1), dtype=np.float64)
    g2_truth = np.ndarray((len(modes),1), dtype=np.float64)
    
    for mode_idx in range(len(modes)):
        
        # e_1
        
        img_space_x = e_1.xyz[0]
        img_space_y = e_1.xyz[1]
        img_space_z = e_1.xyz[2]
        
        beam_space_x_basis = modes[mode_idx]['so3transform'][:,0]
        beam_space_y_basis = modes[mode_idx]['so3transform'][:,1]
        beam_space_z_basis = modes[mode_idx]['so3transform'][:,2]

        beam_space_x = img_space_x * beam_space_x_basis[0] + img_space_y * beam_space_x_basis[1] + img_space_z * beam_space_x_basis[2]
        beam_space_y = img_space_x * beam_space_y_basis[0] + img_space_y * beam_space_y_basis[1] + img_space_z * beam_space_y_basis[2]
        beam_space_z = img_space_x * beam_space_z_basis[0] + img_space_y * beam_space_z_basis[1] + img_space_z * beam_space_z_basis[2]

        beam_space_z -= modes[mode_idx]['offset_z']
        
        w_z = w_0 * np.sqrt(1 + (beam_space_z / z_r)**2)
            
        h_m = modes[mode_idx]['gh_m'](np.sqrt(2) * beam_space_x / w_z)
        h_n = modes[mode_idx]['gh_n'](np.sqrt(2) * beam_space_y / w_z)

        scalar_comp = E_0 * w_0 / w_z * h_m * h_n

        sum_xy_squares = beam_space_x**2 + beam_space_y**2

        inv_w_squared = 1 / w_z**2

        inv_r = beam_space_z / (beam_space_z**2 + z_r**2)

        vergence_comp = 1.j * wavenumber * inv_r / 2

        wavenumber_comp = 1.j * wavenumber * beam_space_z

        gouy_phase_shift = 1.j * (modes[mode_idx]['mn'][0] + modes[mode_idx]['mn'][1] + 1) * np.arctan(beam_space_z / z_r)

        exp_component = np.exp(-sum_xy_squares * (inv_w_squared + vergence_comp) - wavenumber_comp - gouy_phase_shift)
        
        e_1_i = np.abs(scalar_comp * exp_component)**2
        
        e_1_p = e_1.relative_brightness * e_1_i
        
        p_1_true = detector.detectFn(e_1.xyz, e_1_p)
        
        # e_2
        
        img_space_x = e_2.xyz[0]
        img_space_y = e_2.xyz[1]
        img_space_z = e_2.xyz[2]
        
        beam_space_x_basis = modes[mode_idx]['so3transform'][:,0]
        beam_space_y_basis = modes[mode_idx]['so3transform'][:,1]
        beam_space_z_basis = modes[mode_idx]['so3transform'][:,2]

        beam_space_x = img_space_x * beam_space_x_basis[0] + img_space_y * beam_space_x_basis[1] + img_space_z * beam_space_x_basis[2]
        beam_space_y = img_space_x * beam_space_y_basis[0] + img_space_y * beam_space_y_basis[1] + img_space_z * beam_space_y_basis[2]
        beam_space_z = img_space_x * beam_space_z_basis[0] + img_space_y * beam_space_z_basis[1] + img_space_z * beam_space_z_basis[2]

        beam_space_z -= modes[mode_idx]['offset_z']
        
        w_z = w_0 * np.sqrt(1 + (beam_space_z / z_r)**2)
            
        h_m = modes[mode_idx]['gh_m'](np.sqrt(2) * beam_space_x / w_z)
        h_n = modes[mode_idx]['gh_n'](np.sqrt(2) * beam_space_y / w_z)

        scalar_comp = E_0 * w_0 / w_z * h_m * h_n

        sum_xy_squares = beam_space_x**2 + beam_space_y**2

        inv_w_squared = 1 / w_z**2

        inv_r = beam_space_z / (beam_space_z**2 + z_r**2)

        vergence_comp = 1.j * wavenumber * inv_r / 2

        wavenumber_comp = 1.j * wavenumber * beam_space_z

        gouy_phase_shift = 1.j * (modes[mode_idx]['mn'][0] + modes[mode_idx]['mn'][1] + 1) * np.arctan(beam_space_z / z_r)

        exp_component = np.exp(-sum_xy_squares * (inv_w_squared + vergence_comp) - wavenumber_comp - gouy_phase_shift)
        
        e_2_i = np.abs(scalar_comp * exp_component)**2
        
        e_2_p = e_2.relative_brightness * e_2_i
        
        p_2_true = detector.detectFn(e_2.xyz, e_2_p)
        
        # set g1 g2
        
        g1_truth[mode_idx] = (p_1_true + p_2_true) / (e_1.relative_brightness + e_2.relative_brightness)
        
        alpha = p_2_true / p_1_true
        
        g2_truth[mode_idx] = (2 * alpha) / (1 + alpha)**2
        
    print('Truth G1:')
    print(g1_truth)
    
    print('Truth G2:')
    print(g2_truth)
    
    #======================================================================================================================================================================================================
    # Plotting, if we want it
    #======================================================================================================================================================================================================

    for mode_idx in range(len(modes)):
        
        if PLOT_2D:
        
            img_space_y_meshgrid, img_space_x_meshgrid = np.meshgrid(
                x_linspace,
                y_linspace,
            )

            img_space_z_meshgrid = np.zeros_like(img_space_x_meshgrid)
            
            x_basis = np.array([ 1.,0.,0. ], dtype=np.float64)
            y_basis = np.array([ 0.,1.,0. ], dtype=np.float64)
            z_basis = np.array([ 0.,0.,1. ], dtype=np.float64)

            beam_space_x_basis = modes[mode_idx]['so3transform'][:,0]
            beam_space_y_basis = modes[mode_idx]['so3transform'][:,1]
            beam_space_z_basis = modes[mode_idx]['so3transform'][:,2]

            beam_space_x_meshgrid = img_space_x_meshgrid * beam_space_x_basis[0] + img_space_y_meshgrid * beam_space_x_basis[1] + img_space_z_meshgrid * beam_space_x_basis[2]
            beam_space_y_meshgrid = img_space_x_meshgrid * beam_space_y_basis[0] + img_space_y_meshgrid * beam_space_y_basis[1] + img_space_z_meshgrid * beam_space_y_basis[2]
            beam_space_z_meshgrid = img_space_x_meshgrid * beam_space_z_basis[0] + img_space_y_meshgrid * beam_space_z_basis[1] + img_space_z_meshgrid * beam_space_z_basis[2]

            beam_space_z_meshgrid -= modes[mode_idx]['offset_z']

            w_z_meshgrid = w_0 * np.sqrt(1 + (beam_space_z_meshgrid / z_r)**2)
            
            h_m = modes[mode_idx]['gh_m'](np.sqrt(2) * beam_space_x_meshgrid / w_z_meshgrid)
            h_n = modes[mode_idx]['gh_n'](np.sqrt(2) * beam_space_y_meshgrid / w_z_meshgrid)
            
            scalar_comp = E_0 * w_0 / w_z_meshgrid * h_m * h_n

            sum_xy_squares = beam_space_x_meshgrid**2 + beam_space_y_meshgrid**2

            inv_w_squared = 1 / w_z_meshgrid**2

            inv_r = beam_space_z_meshgrid / (beam_space_z_meshgrid**2 + z_r**2)

            vergence_comp = 1.j * wavenumber * inv_r / 2

            wavenumber_comp = 1.j * wavenumber * beam_space_z_meshgrid

            gouy_phase_shift = 1.j * (modes[mode_idx]['mn'][0] + modes[mode_idx]['mn'][1] + 1) * np.arctan(beam_space_z_meshgrid / z_r)

            exp_component = np.exp(-sum_xy_squares * (inv_w_squared + vergence_comp) - wavenumber_comp - gouy_phase_shift)

            intensity = np.abs(scalar_comp * exp_component)**2
            
            intensity /= np.max(intensity)
            
            plt.title(r'$\mathrm{TEM}_{mn},\;m=' + str(modes[mode_idx]['mn'][0]) + r',\;n=' + str(modes[mode_idx]['mn'][1]) + r'$', fontsize=28, pad=10)
            plt.pcolormesh(img_space_x_meshgrid / w_0, img_space_y_meshgrid / w_0, intensity, cmap=parula)
            cbar = plt.colorbar(pad=0.01)
            cbar.ax.tick_params(labelsize=18)
            cbar.set_label(r"$I_{mn}\left(x,y\right)/I_{max}$", fontsize=24, rotation=-90, labelpad=28)
            plt.scatter(e_1.xy[0] / w_0, e_1.xy[1] / w_0, c='r', marker='x', s=40, linewidths=1)
            plt.scatter(e_2.xy[0] / w_0, e_2.xy[1] / w_0, facecolors='none', edgecolors='r', marker='o', s=20, linewidths=1)
            plt.xlabel(r"$x/w$", fontsize=24)
            plt.ylabel(r"$y/w$", fontsize=24)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            # cbar.set_ticks
            plt.gca().set_aspect(1)
            plt.tight_layout()
            plt.savefig(os.path.join(path,'localization-3d-tem-results','image_plane_' + modes[mode_idx]['name'] + '_' + str(modes[mode_idx]['mn'][0]) + '_' + str(modes[mode_idx]['mn'][1]) + '.png'), dpi=400, bbox_inches='tight')
            plt.close()
            
        if PLOT_3D:
            
            x_linspace = np.linspace(-pm_w_0,pm_w_0,x_3d_samples)
            y_linspace = np.linspace(-pm_w_0,pm_w_0,y_3d_samples)
            z_linspace = np.linspace(-pm_z,pm_z,z_3d_samples)
            
            img_space_y_meshgrid, img_space_x_meshgrid, img_space_z_meshgrid = np.meshgrid(x_linspace, y_linspace, z_linspace)
            
            beam_space_x_basis = rotation[:,0]
            beam_space_y_basis = rotation[:,1]
            beam_space_z_basis = rotation[:,2]

            beam_space_x_basis = modes[mode_idx]['so3transform'][:,0]
            beam_space_y_basis = modes[mode_idx]['so3transform'][:,1]
            beam_space_z_basis = modes[mode_idx]['so3transform'][:,2]

            beam_space_x_meshgrid = img_space_x_meshgrid * beam_space_x_basis[0] + img_space_y_meshgrid * beam_space_x_basis[1] + img_space_z_meshgrid * beam_space_x_basis[2]
            beam_space_y_meshgrid = img_space_x_meshgrid * beam_space_y_basis[0] + img_space_y_meshgrid * beam_space_y_basis[1] + img_space_z_meshgrid * beam_space_y_basis[2]
            beam_space_z_meshgrid = img_space_x_meshgrid * beam_space_z_basis[0] + img_space_y_meshgrid * beam_space_z_basis[1] + img_space_z_meshgrid * beam_space_z_basis[2]

            beam_space_z_meshgrid -= modes[mode_idx]['offset_z']

            w_z_meshgrid = w_0 * np.sqrt(1 + (beam_space_z_meshgrid / z_r)**2)
            
            h_m = modes[mode_idx]['gh_m'](np.sqrt(2) * beam_space_x_meshgrid / w_z_meshgrid)
            h_n = modes[mode_idx]['gh_n'](np.sqrt(2) * beam_space_y_meshgrid / w_z_meshgrid)
            
            scalar_comp = E_0 * w_0 / w_z_meshgrid * h_m * h_n

            sum_xy_squares = beam_space_x_meshgrid**2 + beam_space_y_meshgrid**2

            inv_w_squared = 1 / w_z_meshgrid**2

            inv_r = beam_space_z_meshgrid / (beam_space_z_meshgrid**2 + z_r**2)

            vergence_comp = 1.j * wavenumber * inv_r / 2

            wavenumber_comp = 1.j * wavenumber * beam_space_z_meshgrid

            gouy_phase_shift = 1.j * (modes[mode_idx]['mn'][0] + modes[mode_idx]['mn'][1] + 1) * np.arctan(beam_space_z_meshgrid / z_r)

            exp_component = np.exp(-sum_xy_squares * (inv_w_squared + vergence_comp) - wavenumber_comp - gouy_phase_shift)

            intensity = np.abs(scalar_comp * exp_component)**2
            
            intensity /= np.max(intensity)
            
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

            # print(pos)

            cam.position = np.array([
                cam_norm * np.cos(cam_az_rad) * np.cos(cam_el_rad),
                cam_norm * np.sin(cam_az_rad) * np.cos(cam_el_rad),
                cam_norm * np.sin(cam_el_rad)
            ])

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
            mlab.savefig(os.path.join(path,'localization-3d-tem-results','beam_3d_' + modes[mode_idx]['name'] + '_' + str(modes[mode_idx]['mn'][0]) + '_' + str(modes[mode_idx]['mn'][1]) + '.png'))

if __name__ == '__main__':
    
    main()