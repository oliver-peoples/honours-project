from scipy.special import hermite as physicistsHermite
from scipy.special._orthogonal import orthopoly1d
import math
import numpy as np

def temModeFnXY(m: int, n: int, x: float, y: float, I_0: float = 1, w_0: float = 1, w: float = 1) -> float:
    
    return I_0 * (w_0 / w)**2 * (physicistsHermite(m)((2**0.5 * x)/w) * math.exp(-x**2 / w**2))**2 * (physicistsHermite(n)((2**0.5 * y)/w) * math.exp(-y**2 / w**2))**2

def fastTEM(hermite_polynomial_m: orthopoly1d, hermite_polynomial_n: orthopoly1d, x: float, y: float, I_0: float = 1, w_0: float = 1, w: float = 1) -> float:
    
    return I_0 * (w_0 / w)**2 * (hermite_polynomial_m((2**0.5 * x)/w) * math.exp(-x**2 / w**2))**2 * (hermite_polynomial_n((2**0.5 * y)/w) * math.exp(-y**2 / w**2))**2

def parallelTEM(m: int, n: int, y_val: float, grid_x: float, x_range) -> np.ndarray:

    x_linspace = np.linspace(*x_range, grid_x)
    
    intensity_vals = np.ndarray((grid_x), dtype=np.float32)
    
    hermite_polynomial_m = physicistsHermite(m)
    hermite_polynomial_n = physicistsHermite(n)
    
    for x_idx in range(0, grid_x):
            
            # intensity_vals[x_idx] = temModeFnXY(m, n, x_linspace[x_idx], y_val)
            intensity_vals[x_idx] = fastTEM(hermite_polynomial_m, hermite_polynomial_n, x_linspace[x_idx], y_val)
            
    return intensity_vals

def parallelTEM_Affine(m: int, n: int, y_val: float, grid_x: float, translation: np.ndarray, rotation: np.ndarray, x_range) -> np.ndarray:

    x_linspace = np.linspace(*x_range, grid_x)
    
    intensity_vals = np.ndarray((grid_x), dtype=np.float32)
    
    hermite_polynomial_m = physicistsHermite(m)
    hermite_polynomial_n = physicistsHermite(n)
    
    for x_idx in range(0, grid_x):
            
            intensity_vals[x_idx] = fastTEM(hermite_polynomial_m, hermite_polynomial_n, x_linspace[x_idx] - translation[0], y_val - translation[1])
            
    return intensity_vals

def infiniteIntegrals():
    
    return (0.5 * math.sqrt(2) * )

def normalizationCoefficient(m: int, w: float=1):
    
    