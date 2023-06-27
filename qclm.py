from dataclasses import dataclass
import numpy as np
from scipy.special import hermite as physicistsHermite
from scipy.special._orthogonal import orthopoly1d
from typing import List

@dataclass
class Emitter:
    
    xy: np.array
    relative_brightness: float
    
@dataclass
class RectangularTEM:
    
    m: int
    n: int
    w: float
    rot: float
    norm_coeff: float = 1.
    
    def __post_init__(self) -> None:
        
        self.hermite_polynomial_m = physicistsHermite(self.m)
        self.hermite_polynomial_n = physicistsHermite(self.n)
        
def uniformIlluminationScan():
    
    return RectangularTEM(0, 0, 100000, 0.)

@dataclass
class Detector:
    
    xy: np.array = np.array([0.,0.])
    w: float = 1
    
# @dataclass
# class GaussLaguerreTEM:
    
#     p: int
#     l: int
#     rho: float

def temModeFnXY(m: int, n: int, x: float, y: float, I_0: float = 1, w_0: float = 1, w: float = 1) -> float:
    
    return I_0 * (w_0 / w)**2 * (physicistsHermite(m)((2**0.5 * x)/w) * np.exp(-x**2 / w**2))**2 * (physicistsHermite(n)((2**0.5 * y)/w) * np.exp(-y**2 / w**2))**2

def fastTEM(hermite_polynomial_m: orthopoly1d, hermite_polynomial_n: orthopoly1d, x: float, y: float, I_0: float = 1, w_0: float = 1, w: float = 1) -> float:
    
    return I_0 * (w_0 / w)**2 * (hermite_polynomial_m((2**0.5 * x)/w) * np.exp(-x**2 / w**2))**2 * (hermite_polynomial_n((2**0.5 * y)/w) * np.exp(-y**2 / w**2))**2

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

def parallelConfocalScan(emitters: List[Emitter], light_structure: RectangularTEM, y_val, x_linspace, detector_w) -> np.ndarray:
    
    linspace_len = np.size(x_linspace)
    
    g_1_g_2_concatenated = np.ones(2 * linspace_len, dtype=np.float64)
    
    for x_idx in range(0, np.size(x_linspace)):
        
        xy_objective = np.array([x_linspace[x_idx], y_val], dtype=np.float64)
        
        xy_emitter_1_relative = emitters[0].xy - xy_objective
        xy_emitter_2_relative = emitters[1].xy - xy_objective
        
        r_1 = np.linalg.norm(xy_emitter_1_relative, axis=0)
        r_2 = np.linalg.norm(xy_emitter_2_relative, axis=0)
        
        p_1 = emitters[0].relative_brightness * light_structure.w * fastTEM(
            hermite_polynomial_m=light_structure.hermite_polynomial_m,
            hermite_polynomial_n=light_structure.hermite_polynomial_n,
            x=xy_emitter_1_relative[0],
            y=xy_emitter_1_relative[1],
            I_0=1,
            w_0=1,
            w=light_structure.w
        )
        
        p_2 = emitters[1].relative_brightness * light_structure.w * fastTEM(
            hermite_polynomial_m=light_structure.hermite_polynomial_m,
            hermite_polynomial_n=light_structure.hermite_polynomial_n,
            x=xy_emitter_2_relative[0],
            y=xy_emitter_2_relative[1],
            I_0=1,
            w_0=1,
            w=light_structure.w
        )
        
        p_1 = np.exp(-(r_1**2 / 2)/(2 * detector_w**2)) * p_1
        p_2 = np.exp(-(r_2**2 / 2)/(2 * detector_w**2)) * p_2
            
        g_1_g_2_concatenated[x_idx] = (p_1 + p_2) / (emitters[0].relative_brightness + emitters[1].relative_brightness)
        
        alpha = p_1 / p_2
        
        g_1_g_2_concatenated[linspace_len + x_idx] = (2 * alpha) / (1 + alpha)**2
        
    return g_1_g_2_concatenated

def groundTruthG1_G2(detector: Detector, emitters: List[Emitter], light_structures: List[RectangularTEM], xy_objective):
    
    # xy_objective = np.array([0.,0.], dtype=np.float64)
        
    xy_emitter_1_relative = emitters[0].xy - xy_objective
    xy_emitter_2_relative = emitters[1].xy - xy_objective
    
    r_1 = np.linalg.norm(xy_emitter_1_relative, axis=0)
    r_2 = np.linalg.norm(xy_emitter_2_relative, axis=0)
    
    g_1_pred = np.ndarray((len(light_structures),1), dtype=np.float64)
    g_2_pred = np.ndarray((len(light_structures),1), dtype=np.float64)
    
    for light_structure_idx in range(len(light_structures)):
        
        light_structure = light_structures[light_structure_idx]
        
        # hermite_polynomial_m = physicistsHermite(light_structure[0][0])
        # hermite_polynomial_n = physicistsHermite(light_structure[0][1])
        
        p_1 = emitters[0].relative_brightness * light_structure.norm_coeff * fastTEM(
            hermite_polynomial_m=light_structure.hermite_polynomial_m,
            hermite_polynomial_n=light_structure.hermite_polynomial_n,
            x=xy_emitter_1_relative[0],
            y=xy_emitter_1_relative[1],
            I_0=1,
            w_0=1,
            w=light_structure.w
        )
        
        p_2 = emitters[1].relative_brightness * light_structure.norm_coeff * fastTEM(
            hermite_polynomial_m=light_structure.hermite_polynomial_m,
            hermite_polynomial_n=light_structure.hermite_polynomial_n,
            x=xy_emitter_2_relative[0],
            y=xy_emitter_2_relative[1],
            I_0=1,
            w_0=1,
            w=light_structure.w
        )
        
        p_1 = np.exp(-(r_1**2 / 2)/(2 * detector.w**2)) * p_1
        p_2 = np.exp(-(r_2**2 / 2)/(2 * detector.w**2)) * p_2
            
        g_1_pred[light_structure_idx] = (p_1 + p_2) / (emitters[0].relative_brightness + emitters[1].relative_brightness)
        
        alpha = p_2 / p_1
        
        g_2_pred[light_structure_idx] = (2 * alpha) / (1 + alpha)**2
        
    return g_1_pred,g_2_pred

def optimizeMe(ps_vec, detector: Detector, emitters: List[Emitter], light_structures: List[RectangularTEM], xy_objective, noisy_g_1, noisy_g_2):
    
    p_0_2 = ps_vec[4]
    
    xy_emitter_1 = Emitter(ps_vec[0:2], 1.0)
    xy_emitter_2 = Emitter(ps_vec[2:4], p_0_2)
    
    g_1_pred = np.ndarray((len(light_structures),1), dtype=np.float64)
    g_2_pred = np.ndarray((len(light_structures),1), dtype=np.float64)
    
    for light_structure_idx in range(len(light_structures)):
        
        light_structure = light_structures[light_structure_idx]
        
        # hermite_polynomial_m = physicistsHermite(light_structure[0][0])
        # hermite_polynomial_n = physicistsHermite(light_structure[0][1])
        
        xy_emitter_1_relative = xy_emitter_1.xy - xy_objective
        xy_emitter_2_relative = xy_emitter_2.xy - xy_objective
        
        r_1 = np.linalg.norm(xy_emitter_1_relative, axis=0)
        r_2 = np.linalg.norm(xy_emitter_2_relative, axis=0)
        
        p_1 = emitters[0].relative_brightness * light_structure.norm_coeff * fastTEM(
            hermite_polynomial_m=light_structure.hermite_polynomial_m,
            hermite_polynomial_n=light_structure.hermite_polynomial_n,
            x=xy_emitter_1_relative[0],
            y=xy_emitter_1_relative[1],
            I_0=1,
            w_0=1,
            w=light_structure.w
        )
        
        p_2 = emitters[1].relative_brightness * light_structure.norm_coeff * fastTEM(
            hermite_polynomial_m=light_structure.hermite_polynomial_m,
            hermite_polynomial_n=light_structure.hermite_polynomial_n,
            x=xy_emitter_2_relative[0],
            y=xy_emitter_2_relative[1],
            I_0=1,
            w_0=1,
            w=light_structure.w
        )
        
        p_1 = np.exp(-(r_1**2 / 2)/(2 * detector.w**2)) * p_1
        p_2 = np.exp(-(r_2**2 / 2)/(2 * detector.w**2)) * p_2
        
        g_1_pred[light_structure_idx] = (p_1 + p_2) / (1. + p_0_2)
        
        alpha = p_2 / p_1
                
        g_2_pred[light_structure_idx] = (2 * alpha) / (1 + alpha)**2
        
    return np.sum((g_1_pred - noisy_g_1)**2) + np.sum((g_2_pred - noisy_g_2)**2)