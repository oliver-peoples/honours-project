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
    
    grid_x = 1750
    grid_y = 1750
    
    x_range = (-3.0, 3.0)
    y_range = (-3.0, 3.0)
    
    y_linspace = np.linspace(*y_range, grid_y)
    x_linspace = np.linspace(*x_range, grid_x)
    
    for m in range(0,4):
        
        for n in range(0,4):
            
            print(m, n)
    
            arg_vec = [(m, n, y_val, grid_x, x_range) for y_val in y_linspace]
            
            multiprocessing_pool = mp.Pool(processes=mp.cpu_count())
            
            intensity_vals = np.stack(multiprocessing_pool.starmap(parallelTEM, arg_vec), axis=0)
            
            volume = np.trapz(
                y=np.asarray(
                    [np.trapz(y=intensity_row, x=x_linspace) for intensity_row in intensity_vals[:]]
                ),
                x=y_linspace
            )
            
            print(volume)
            
            intensity_vals *= 1. / volume
                    
            exec('plt.title(r"$\mathbf{TEM}_{' + str(m) + ',' + str(n) + '}$", fontsize=16)')
            plt.pcolormesh(x_linspace / w, y_linspace / w, intensity_vals / np.max(intensity_vals), cmap='Blues')
            plt.xlabel(r"$x/w$", fontsize=16)
            plt.ylabel(r"$y/w$", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            cbar = plt.colorbar()
            cbar.set_label(r"$P_{x,y}/P_{max}$", fontsize=16, rotation=-90, labelpad=20)
            # cbar.set_ticks
            plt.gca().set_aspect(1)
            plt.tight_layout()
            plt.savefig(f'tem_m_{m}_n_{n}.png', dpi=1200, bbox_inches='tight')
            plt.close()
            
            np.savetxt(f'tem_m_{m}_n_{n}.csv', intensity_vals, delimiter=',')

if __name__ == "__main__":
    
    main()