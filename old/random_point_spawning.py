from scipy.special import hermite as physicistsHermite
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing as mp
from tem_utils import *

matplotlib.rcParams['text.usetex'] = True

def main() -> None:
    
    w = 1.
    
    grid_x = 750
    grid_y = 750
    
    x_range = (-2.0, 2.0)
    y_range = (-2.0, 2.0)
    
    y_linspace = np.linspace(*y_range, grid_y)
    x_linspace = np.linspace(*x_range, grid_x)
    
    emitter_locations = np.array([
        [-1. / w,-1.5 / w],
        [1.0 / w,1.0 / w],
        [0.5 / w,-1.5 / w]
    ], dtype=np.float32)
    
    emitter_arrays = []
    
    emitter_idx = [ 0,1,2 ]
    
    multiprocessing_pool = mp.Pool(processes=mp.cpu_count())
    
    rotation_matrix = np.eye(2,2)
    
    for emitter_data in zip(emitter_locations[:], emitter_idx):
        
        arg_vec = [(0, 0, y_val, grid_x, emitter_data[0], rotation_matrix, x_range) for y_val in y_linspace]
        
        intensity_vals = np.stack(multiprocessing_pool.starmap(parallelTEM_Affine, arg_vec), axis=0)
        
        intensity_vals *= 1. / np.max(intensity_vals)
                
        exec('plt.title(r"$\mathbf{Emitter\;' + str(emitter_data[1]) + '}$", fontsize=16)')
        plt.pcolormesh(x_linspace / w, y_linspace / w, intensity_vals, cmap='Blues')
        plt.xlabel(r"$x/w$", fontsize=16)
        plt.ylabel(r"$y/w$", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        cbar = plt.colorbar()
        cbar.set_label(r"$P_{x,y}/P_{max}$", fontsize=16, rotation=-90, labelpad=20)
        # cbar.set_ticks
        plt.gca().set_aspect(1)
        plt.tight_layout()
        plt.savefig(f'emitter_{emitter_data[1]}.png', dpi=1200, bbox_inches='tight')
        plt.close()
        
        emitter_arrays.append(intensity_vals)
    
    for m in range(0,4):
        
        for n in range(0,4):
    
            arg_vec = [(m, n, y_val, grid_x, x_range) for y_val in y_linspace]
            
            multiprocessing_pool = mp.Pool(processes=mp.cpu_count())
            
            excitation_psf = np.stack(multiprocessing_pool.starmap(parallelTEM, arg_vec), axis=0)
            
            for emitter_array in emitter_arrays:
                
                excitation_psf = np.multiply(excitation_psf, emitter_array)
            
            total_psf = excitation_psf / np.max(excitation_psf)
                    
            exec('plt.title(r"$\mathbf{TEM}_{' + str(m) + ',' + str(n) + '}\mathbf{\;With\;Emitters}$", fontsize=16)')
            plt.pcolormesh(x_linspace / w, y_linspace / w, total_psf, cmap='Blues')
            plt.xlabel(r"$x/w$", fontsize=16)
            plt.ylabel(r"$y/w$", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            cbar = plt.colorbar()
            cbar.set_label(r"$P_{x,y}/P_{max}$", fontsize=16, rotation=-90, labelpad=20)
            # cbar.set_ticks
            plt.gca().set_aspect(1)
            plt.tight_layout()
            plt.savefig(f'emitters_with_tem_{m}_n_{n}.png', dpi=1200, bbox_inches='tight')
            plt.close()
            
            np.savetxt(f'emitters_with_tem_{m}_n_{n}.csv', intensity_vals, delimiter=',')

if __name__ == "__main__":
    
    main()