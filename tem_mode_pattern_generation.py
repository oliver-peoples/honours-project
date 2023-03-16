from scipy.special import hermite as physicistsHermite
import math
import numpy as np
import matplotlib.pyplot as plt

def temModeFnXY(m: int, n: int, x: float, y: float, I_0: float = 1, w_0: float = 1, w: float = 1) -> float:
    
    return I_0 * (w_0 / w)**2 * (physicistsHermite(m)((2**0.5 * x)/w) * math.exp(-x**2 / -w**2))**2 * (physicistsHermite(n)((2**0.5 * y)/w) * math.exp(-y**2 / -w**2))**2

def main() -> None:
    
    grid_x = 100
    grid_y = 100
    
    m = 1
    n = 2
    
    x_range = (-1.0, 1.0)
    y_range = (-1.0, 1.0)
    
    x_linspace = np.linspace(*x_range, grid_x)
    y_linspace = np.linspace(*y_range, grid_y)
    
    intensity_vals = np.ndarray((grid_x, grid_y), dtype=np.float32)
    
    for y_idx in range(0, grid_y):
        
        for x_idx in range(0, grid_x):
            
            intensity_vals[y_idx,x_idx] = temModeFnXY(m, n, x_linspace[x_idx], y_linspace[y_idx])
            
    plt.imshow(intensity_vals)
    plt.show()

if __name__ == "__main__":
    
    main()