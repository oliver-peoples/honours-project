from scipy.special import hermite as physicistsHermite
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing as mp
from tem_utils import *
import parula

matplotlib.rcParams['text.usetex'] = True

def main() -> None:
    
    w = 2.
    
    grid_x = 1000
    grid_y = 1000
    
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
                    
            plt.title(r'$\mathrm{m}=' + str(m) + r',\mathrm{n}=' + str(n) + r'$', fontsize=28)
            plt.pcolormesh(x_linspace / w, y_linspace / w, intensity_vals / np.max(intensity_vals), cmap=parula.parula)
            plt.xlabel(r"$x/w$", fontsize=24)
            plt.ylabel(r"$y/w$", fontsize=24)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            cbar = plt.colorbar(pad=0.01)
            cbar.ax.tick_params(labelsize=18)
            cbar.set_label(r"$I\left(x,y\right)/I_{max}$", fontsize=24, rotation=-90, labelpad=28)
            # cbar.set_ticks
            plt.gca().set_aspect(1)
            plt.tight_layout()
            plt.savefig(f'tem-modes/tem_m_{m}_n_{n}.png', dpi=500, bbox_inches='tight')
            plt.close()
            
            # np.savetxt(f'tem_m_{m}_n_{n}.csv', intensity_vals, delimiter=',')

if __name__ == "__main__":
    
    main()