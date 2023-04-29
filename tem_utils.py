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

def parallelConfocalScan(emitters, light_structure, y_val, x_linspace, detector_w) -> np.ndarray:
    
    linspace_len = np.size(x_linspace)
    
    g_1_g_2_concatenated = np.ones(2 * linspace_len, dtype=np.float64)
    
    hermite_polynomial_m = physicistsHermite(light_structure[0][0])
    hermite_polynomial_n = physicistsHermite(light_structure[0][1])
    
    for x_idx in range(0, np.size(x_linspace)):
        
        xy_objective = np.array([x_linspace[x_idx], y_val], dtype=np.float64)
        
        xy_emitter_1_relative = emitters[0][0] - xy_objective
        xy_emitter_2_relative = emitters[1][0] - xy_objective
        
        r_1 = np.linalg.norm(xy_emitter_1_relative, axis=0)
        r_2 = np.linalg.norm(xy_emitter_2_relative, axis=0)
        
        p_1 = emitters[0][1] * light_structure[2] * fastTEM(
            hermite_polynomial_m=hermite_polynomial_m,
            hermite_polynomial_n=hermite_polynomial_n,
            x=xy_emitter_1_relative[0],
            y=xy_emitter_1_relative[1],
            I_0=1,
            w_0=1,
            w=light_structure[1]
        )
        
        p_2 = emitters[1][1] * light_structure[2] * fastTEM(
            hermite_polynomial_m=hermite_polynomial_m,
            hermite_polynomial_n=hermite_polynomial_n,
            x=xy_emitter_2_relative[0],
            y=xy_emitter_2_relative[1],
            I_0=1,
            w_0=1,
            w=light_structure[1]
        )
        
        p_1 = np.exp(-(r_1**2 / 2)/(2 * detector_w**2)) * p_1
        p_2 = np.exp(-(r_2**2 / 2)/(2 * detector_w**2)) * p_2
            
        g_1_g_2_concatenated[x_idx] = (p_1 + p_2) / (emitters[0][1] + emitters[1][1])
        
        alpha = p_1 / p_2
        
        g_1_g_2_concatenated[linspace_len + x_idx] = (2 * alpha) / (1 + alpha)**2
        
    return g_1_g_2_concatenated

def sCoefficient(m: int) -> float:
    
    return (0.5,1,4,24)[m]

def normalizationCoefficient(m: int, n: int, w: float):
    
    return 1. / (2. * np.pi * w**2 * sCoefficient(m) * sCoefficient(n))