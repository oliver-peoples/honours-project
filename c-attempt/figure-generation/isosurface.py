import numpy as np
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
import cv2
import glob

import os

path = os.path.dirname(__file__)

import sys

sys.path.append(os.path.abspath(os.path.join(path,os.path.pardir)))

from parula import parula
from qclm import genHermite, genLaguerre, mlab_imshowColor
from scipy.optimize import minimize
from scipy.optimize import fmin

E_0 = 1.

GH = 1
GL = 2

TEM_MODE = GL

def main() -> None:
    
    if TEM_MODE == GH:
        
        ghPlot()
        
    else:
            # if TEM_MODE == GH:
        
        glPlot()
        
    # else:
        # 
        # glPlot()
        # glPlot()
    
    pic_paths = glob.glob(os.path.join(path,'isosurfaces','*.png'))
    
    for pic_path in pic_paths:
        
        if 'new' not in pic_path:
        
            img = cv2.imread(pic_path)
            
            img = img[10:1850,400:1600]
            
            cv2.imwrite(pic_path.split('.png')[0] + '_new.png', img)
            
            middle_idx = np.shape(img)
            print(middle_idx)
        # cv2.imshow('test',img)
        # cv2.waitKey(0)
        
def glPlot() -> None:
    
    p = 0
    l = 0

    gl_poly = genLaguerre(p,l)

    wavelength = 632.8

    w_0 = 400.0
    n_s = 1.

    x_slice = 0. 

    wavelength *= 1e-9
    wavenumber = 2 * np.pi / wavelength
    w_0 *= 1e-9

    z_r = np.pi * w_0**2 * n_s / wavelength

    pm_z = z_r * 3

    pm_w_0 = w_0 * 3.

    x_samples = 300
    y_samples = 300
    z_samples = 300

    off_nadir_angle_deg = 0
    azimuth_deg = 0

    # roll_linspace = np.linspace(0,90,300)

    # for roll_idx in range(len(roll_linspace)):

    roll_deg = 0

    z_offset = 0. * wavelength

    x_linspace = np.linspace(-pm_w_0,pm_w_0,x_samples)
    y_linspace = np.linspace(-pm_w_0,pm_w_0,y_samples)
    z_linspace = np.linspace(-pm_z,pm_z,z_samples)

    normal = np.array([0.,0.,1.], dtype=np.float64).T

    img_space_y_meshgrid, img_space_x_meshgrid, img_space_z_meshgrid = np.meshgrid(x_linspace, y_linspace, z_linspace)

    y_rotation = Rotation.from_rotvec(off_nadir_angle_deg * np.pi / 180 * np.array([0,1,0], dtype=np.float64))
    z_rotation = Rotation.from_rotvec(azimuth_deg * np.pi / 180 * np.array([0,0,1], dtype=np.float64))

    rotation = z_rotation.as_matrix() @ y_rotation.as_matrix()

    normal = rotation @ normal

    roll_rotation = Rotation.from_rotvec(roll_deg * np.pi / 180 * normal).as_matrix()

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
    
    beam_space_r_meshgrid = np.sqrt(beam_space_x_meshgrid**2 + beam_space_y_meshgrid**2)
    beam_space_psi_meshgrid = np.arctan2(beam_space_y_meshgrid, beam_space_x_meshgrid)
                
    w_z_meshgrid = w_0 * np.sqrt(1 + (beam_space_z_meshgrid / z_r)**2)
    
    # inv_w_squared = 1 / w_z_meshgrid**2

    inv_2r = 0.5 * beam_space_z_meshgrid / (beam_space_z_meshgrid**2 + z_r**2)
    
    gouy_phase_shift = 1.j * (l + 2*p + 1) * np.arctan(beam_space_z_meshgrid / z_r)
    
    comp_1 = (beam_space_r_meshgrid * np.sqrt(2)/w_z_meshgrid)**l
    comp_2 = np.exp(-beam_space_r_meshgrid**2 / w_z_meshgrid**2)
    comp_3 = gl_poly(2 * beam_space_r_meshgrid**2 / w_z_meshgrid**2)
    comp_4 = np.exp(-1.j * wavenumber * beam_space_r_meshgrid**2 * inv_2r)
    comp_5 = np.exp(-1.j * l * beam_space_psi_meshgrid)
    comp_6 = np.exp(gouy_phase_shift)
    
    E_field = E_0 * comp_1 * comp_2 * comp_3 * comp_4 * comp_5 * comp_6
    
    C_lg = np.sqrt(2 * np.math.factorial(p) / (np.pi * np.math.factorial(p + l)))

    intensity = np.abs(C_lg * E_field)**2

    max_intensity = np.max(intensity)
    
    print(max_intensity)
    
    first_thresh = 0.75 * max_intensity
    second_thresh = 0.5 * max_intensity
    third_thresh = 0.15 * max_intensity
    
    print(f'1st: {first_thresh}, 2nd: {second_thresh}, 3rd: {third_thresh}')
    # intensity /=max_intensity

    fig = mlab.figure(size=(2000,2000), bgcolor=(1,1,1))

    src = mlab.pipeline.scalar_field(img_space_x_meshgrid * 1e9, img_space_y_meshgrid * 1e9, img_space_z_meshgrid * 1e9, intensity)
    mlab.pipeline.iso_surface(src, contours=[first_thresh], opacity=1, color=parula(0.75)[:3])
    mlab.pipeline.iso_surface(src, contours=[second_thresh], opacity=0.5, color=parula(0.5)[:3])
    mlab.pipeline.iso_surface(src, contours=[third_thresh], opacity=0.25, color=parula(0.15)[:3])
    
    cam_az = 45
    cam_el = 30

    cam_az_rad = np.pi * cam_az / 180
    cam_el_rad = np.pi * cam_el / 180

    cam = mlab.gcf().scene.camera
    pos = cam.position
    scale = 0.75
    cam_norm = scale * np.linalg.norm(cam.position)

    print(pos)

    cam.position = np.array([
        cam_norm * np.cos(cam_az_rad) * np.cos(cam_el_rad),
        cam_norm * np.sin(cam_az_rad) * np.cos(cam_el_rad),
        cam_norm * np.sin(cam_el_rad)
    ])

    # 2d slice

    img_space_y_meshgrid, img_space_x_meshgrid = np.meshgrid(
        x_linspace,
        y_linspace,
    )

    img_space_z_meshgrid = np.zeros_like(img_space_x_meshgrid)

    beam_space_x_meshgrid = img_space_x_meshgrid * beam_space_x_basis[0] + img_space_y_meshgrid * beam_space_x_basis[1] + img_space_z_meshgrid * beam_space_x_basis[2]
    beam_space_y_meshgrid = img_space_x_meshgrid * beam_space_y_basis[0] + img_space_y_meshgrid * beam_space_y_basis[1] + img_space_z_meshgrid * beam_space_y_basis[2]
    beam_space_z_meshgrid = img_space_x_meshgrid * beam_space_z_basis[0] + img_space_y_meshgrid * beam_space_z_basis[1] + img_space_z_meshgrid * beam_space_z_basis[2]

    beam_space_z_meshgrid -= z_offset
    
    beam_space_r_meshgrid = np.sqrt(beam_space_x_meshgrid**2 + beam_space_y_meshgrid**2)
    beam_space_psi_meshgrid = np.arctan2(beam_space_y_meshgrid, beam_space_x_meshgrid)
                
    w_z_meshgrid = w_0 * np.sqrt(1 + (beam_space_z_meshgrid / z_r)**2)
    
    inv_w_squared = 1 / w_z_meshgrid**2

    inv_2r = 0.5 * beam_space_z_meshgrid / (beam_space_z_meshgrid**2 + z_r**2)
    
    gouy_phase_shift = (l + 2*p + 1) * np.arctan(beam_space_z_meshgrid / z_r)
    
    comp_1 = (beam_space_r_meshgrid * np.sqrt(2)/w_z_meshgrid)**l
    comp_2 = np.exp(-beam_space_r_meshgrid**2 / w_z_meshgrid**2)
    comp_3 = gl_poly(2 * beam_space_r_meshgrid**2 / w_z_meshgrid**2)
    comp_4 = np.exp(-1.j * wavenumber * beam_space_r_meshgrid**2 * inv_2r)
    comp_5 = np.exp(-1.j * l * beam_space_psi_meshgrid)
    comp_6 = np.exp(1.j * gouy_phase_shift)
    
    E_field = E_0 * comp_1 * comp_2 * comp_3 * comp_4 * comp_5 * comp_6
    
    C_lg = np.sqrt(2 * np.math.factorial(p) / (np.pi * np.math.factorial(p + l)))

    intensity = np.abs(C_lg * E_field)**2

    intensity /= max_intensity

    s = parula(intensity)

    s *= 255

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

    mlab_imshowColor(s[:, :, :3], 255, extent=[-pm_w_0 * 1e9,pm_w_0 * 1e9,-pm_w_0 * 1e9,pm_w_0 * 1e9,0,0])

    f = mlab.gcf()
    f.scene._lift()
    # img_array = mlab.screenshot(figure=f, mode='rgba')
    if ( p+l == 0 ):

        mlab.savefig(os.path.join(path,'isosurfaces',f'tem_gaussian.png'))

    else:
    
        mlab.savefig(os.path.join(path,'isosurfaces',f'tem_p_{p}_l_{l}.png'))
        
def ghPlot() -> None:

    m = 0
    n = 0
    
    norm_factor = pow(2.,-(n+m)/2.) * np.sqrt(2. / (np.pi * np.math.factorial(n) * np.math.factorial(m)))

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

    x_samples = 150
    y_samples = 150
    z_samples = 300

    off_nadir_angle_deg = 0
    azimuth_deg = 0

    # roll_linspace = np.linspace(0,90,300)

    # for roll_idx in range(len(roll_linspace)):

    roll_deg = 0.

    z_offset = 0.0 * w_0

    x_linspace = np.linspace(-pm_w_0,pm_w_0,x_samples)
    y_linspace = np.linspace(-pm_w_0,pm_w_0,y_samples)
    z_linspace = np.linspace(-pm_z,pm_z,z_samples)

    normal = np.array([0.,0.,1.], dtype=np.float64).T

    img_space_y_meshgrid, img_space_x_meshgrid, img_space_z_meshgrid = np.meshgrid(x_linspace, y_linspace, z_linspace)

    y_rotation = Rotation.from_rotvec(off_nadir_angle_deg * np.pi / 180 * np.array([0,1,0], dtype=np.float64))
    z_rotation = Rotation.from_rotvec(azimuth_deg * np.pi / 180 * np.array([0,0,1], dtype=np.float64))

    rotation = z_rotation.as_matrix() @ y_rotation.as_matrix()

    normal = rotation @ normal

    roll_rotation = Rotation.from_rotvec(roll_deg * np.pi / 180 * normal).as_matrix()

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

    intensity = np.abs(norm_factor * scalar_comp * exp_component)**2

    max_intensity = np.max(intensity)
    
    first_thresh = 0.75 * max_intensity
    second_thresh = 0.5 * max_intensity
    third_thresh = 0.15 * max_intensity
    
    print(f'1st: {first_thresh}, 2nd: {second_thresh}, 3rd: {third_thresh}')
    # intensity /=max_intensity

    fig = mlab.figure(size=(2000,2000), bgcolor=(1,1,1))

    src = mlab.pipeline.scalar_field(img_space_x_meshgrid * 1e9, img_space_y_meshgrid * 1e9, img_space_z_meshgrid * 1e9, intensity)
    mlab.pipeline.iso_surface(src, contours=[first_thresh], opacity=1, color=parula(0.75)[:3])
    mlab.pipeline.iso_surface(src, contours=[second_thresh], opacity=0.5, color=parula(0.5)[:3])
    mlab.pipeline.iso_surface(src, contours=[third_thresh], opacity=0.25, color=parula(0.15)[:3])

    cam_az = 45
    cam_el = 30

    cam_az_rad = np.pi * cam_az / 180
    cam_el_rad = np.pi * cam_el / 180

    cam = mlab.gcf().scene.camera
    pos = cam.position
    scale = 0.75
    cam_norm = scale * np.linalg.norm(cam.position)

    print(pos)

    cam.position = np.array([
        cam_norm * np.cos(cam_az_rad) * np.cos(cam_el_rad),
        cam_norm * np.sin(cam_az_rad) * np.cos(cam_el_rad),
        cam_norm * np.sin(cam_el_rad)
    ])

    # 2d slice

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

    intensity = np.abs(norm_factor * scalar_comp * exp_component)**2

    intensity /= max_intensity

    s = parula(intensity)

    s *= 255

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

    mlab_imshowColor(s[:, :, :3], 255, extent=[-pm_w_0 * 1e9,pm_w_0 * 1e9,-pm_w_0 * 1e9,pm_w_0 * 1e9,0,0])

    f = mlab.gcf()
    f.scene._lift()
    # img_array = mlab.screenshot(figure=f, mode='rgba')
    if ( m+n == 0 ):

        mlab.savefig(os.path.join(path,'isosurfaces',f'tem_gaussian.png'))

    else:
    
        mlab.savefig(os.path.join(path,'isosurfaces',f'tem_m_{m}_n_{n}.png'))
    
if __name__ == '__main__':
    
    main()