import numpy as np
import pyvista as pv
import skimage.measure as measure
import os
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import KDTree
from mayavi import mlab
from skimage import measure
from parula import parula

path = os.path.dirname(__file__)

x_basis = np.array([ 1.,0.,0. ], dtype=np.float64)
y_basis = np.array([ 0.,1.,0. ], dtype=np.float64)
z_basis = np.array([ 0.,0.,1. ], dtype=np.float64)

def main() -> None:
    
    meta = np.genfromtxt(os.path.join(path,'meta.out'), delimiter=',')
    
    num_x, num_y, num_z = meta[0,:]
    
    num_x = int(num_x)
    num_y = int(num_y)
    num_z = int(num_z)
    
    bl_corner = meta[1,:]
    tr_corner = meta[2,:]
    
    x_linspace = np.linspace(bl_corner[0], tr_corner[0], num_x)
    y_linspace = np.linspace(bl_corner[1], tr_corner[1], num_y)
    z_linspace = np.linspace(bl_corner[2], tr_corner[2], num_z)

    y_meshgrid, x_meshgrid, z_meshgrid = np.meshgrid(x_linspace, y_linspace, z_linspace)
    
    intensity_field = np.fromfile(os.path.join(path,'tem_intensitys.bin'), np.float64, count=num_x * num_y * num_z)
    
    intensity_field = intensity_field.reshape((num_x, num_y, num_z))
    
    max_intensity = np.max(intensity_field)
    
    intensity /= max_intensity

    fig = mlab.figure(size=(2000,2000), bgcolor=(1,1,1))

    src = mlab.pipeline.scalar_field(x_meshgrid, y_meshgrid, z_meshgrid, intensity)
    mlab.pipeline.iso_surface(src, contours=[0.75], opacity=1, color=parula(0.75)[:3])
    mlab.pipeline.iso_surface(src, contours=[0.5], opacity=0.5, color=parula(0.5)[:3])
    # mlab.pipeline.iso_surface(src, contours=[0.15], opacity=0.25, color=parula(0.15)[:3])

    x_axis = np.array([
        [ 0,0,0 ],
        [*(x_basis * 0.1)]
    ])

    mlab.plot3d(
        x_axis[:,0],
        x_axis[:,1],
        x_axis[:,2],
        color=(1.0,0.0,0.0),
        tube_radius=0.001,
        tube_sides=30
    )

    y_axis = np.array([
        [ 0,0,0 ],
        [*(y_basis * 0.1)]
    ])

    mlab.plot3d(
        y_axis[:,0],
        y_axis[:,1],
        y_axis[:,2],
        color=(0.0,1.0,0.0),
        tube_radius=0.001,
        tube_sides=30
    )

    z_axis = np.array([
        [ 0,0,0 ],
        [*(z_basis * 0.1)]
    ])
    
    mlab.plot3d(
        z_axis[:,0],
        z_axis[:,1],
        z_axis[:,2],
        color=(0.0,1.0,1.0),
        tube_radius=0.001,
        tube_sides=30
    )
    
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
    
    mlab.show()
    
if __name__ == '__main__':
    
    main()