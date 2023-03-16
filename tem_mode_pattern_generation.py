from scipy.special import hermite as physicistsHermite
import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

def temModeFnXY(m: int, n: int, x: float, y: float, I_0: float = 1, w_0: float = 1, w: float = 1) -> float:
    
    return I_0 * (w_0 / w)**2 * (physicistsHermite(m)((2**0.5 * x)/w) * math.exp(-x**2 / w**2))**2 * (physicistsHermite(n)((2**0.5 * y)/w) * math.exp(-y**2 / w**2))**2

def parallelTEM(m: int, n: int, y_val: float, grid_x: float, x_range) -> np.ndarray:

    x_linspace = np.linspace(*x_range, grid_x)
    
    intensity_vals = np.ndarray((grid_x), dtype=np.float32)
    
    for x_idx in range(0, grid_x):
            
            intensity_vals[x_idx] = temModeFnXY(m, n, x_linspace[x_idx] - 1.0, y_val)
            
    return intensity_vals

def main() -> None:
    
    w = 1.
    
    grid_x = 500
    grid_y = 500
    
    m = 1
    n = 2
    
    x_range = (-2.0, 2.0)
    y_range = (-2.0, 2.0)
    
    y_linspace = np.linspace(*y_range, grid_y)
    x_linspace = np.linspace(*x_range, grid_x)
    
    arg_vec = [(m, n, y_val, grid_x, x_range) for y_val in y_linspace]
    
    multiprocessing_pool = mp.Pool(processes=mp.cpu_count())
    
    intensity_vals = np.stack(multiprocessing_pool.starmap(parallelTEM, arg_vec), axis=0)
    
    volume = np.trapz(
        y=np.asarray(
            [np.trapz(y=intensity_row, x=x_linspace) for intensity_row in intensity_vals[:]]
        ),
        x=y_linspace
    )
    
    print(f'volume: {volume}')
    
    intensity_vals *= 1. / volume
            
    plt.pcolormesh(x_linspace / w, y_linspace / w, intensity_vals, cmap='Blues')
    plt.xlabel()
    cbar = plt.colorbar()
    cbar.set_label(r"something cool", fontsize=16, rotation=-90, labelpad=0.5)
    plt.show()

if __name__ == "__main__":
    
    main()