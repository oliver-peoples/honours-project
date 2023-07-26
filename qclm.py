from dataclasses import dataclass
import numpy as np
from scipy.special import genlaguerre as genLaguerre
from scipy.special import hermite as genHermite
from scipy.special._orthogonal import orthopoly1d
from typing import List, Any


@dataclass
class GaussLaguerre:
    
    p: int
    l: int
    
    I_0: float
    waist: float
    
    center: np.ndarray = np.array([0.,0.])
    
    rotation: float = 0
    
    volume: float = 1.
    
    def __post_init__(self) -> None:
        
        self.laguerre = genLaguerre(self.p, self.l)
        
        grid_x = 3750
        grid_y = 3750
        
        x_linspace = np.linspace(-10, 10, grid_x)
        y_linspace = np.linspace(-10, 10, grid_y)
        
        x_meshgrid, y_meshgrid = np.meshgrid(
            x_linspace,
            y_linspace
        )
        
        intensity_map = self.intensityMap(x_meshgrid, y_meshgrid)
        
        volume = np.trapz(
            y=np.asarray(
                [np.trapz(y=intensity_row, x=x_linspace) for intensity_row in intensity_map[:]]
            ),
            x=y_linspace
        )
        
        self.volume = volume
        
    def orderString(self) -> str:
        
        return str(self.p) + str(self.l)
    
    def modeTypeString(self) -> str:
        
        return 'GL'
        
    def rhoFn(self, r) -> float:
    
        return 2. * r**2 / self.waist**2
        
    def intensityFn(self, x: float, y: float):
        
        relative_x = x - self.center[0]
        relative_y = y - self.center[1]
        
        r = np.sqrt(relative_x**2 + relative_y**2)
        phi = np.arctan2(relative_y, relative_x)
        phi += self.rotation
        
        rho = self.rhoFn(r)
        
        gl_tem = self.I_0 * (rho**self.l) * (self.laguerre(rho)**2) * (np.cos(self.l * phi)**2) * np.exp(-rho)
        
        return gl_tem / self.volume
    
    def intensityMap(self, x_meshgrid, y_meshgrid):
        
        return self.intensityFn(x_meshgrid, y_meshgrid)
    
@dataclass
class GaussHermite:
    
    m: int
    n: int
    
    I_0: float
    waist: float
    
    center: np.ndarray = np.array([0.,0.])
    
    rotation: float = 0.
    
    volume: float = 1.
    
    def __post_init__(self) -> None:
        
        self.gh_m = genHermite(self.m)
        self.gh_n = genHermite(self.n)
        
        grid_x = 3750
        grid_y = 3750
        
        x_linspace = np.linspace(-10, 10, grid_x)
        y_linspace = np.linspace(-10, 10, grid_y)
        
        x_meshgrid, y_meshgrid = np.meshgrid(
            x_linspace,
            y_linspace
        )
        
        intensity_map = self.intensityMap(x_meshgrid, y_meshgrid)
        
        volume = np.trapz(
            y=np.asarray(
                [np.trapz(y=intensity_row, x=x_linspace) for intensity_row in intensity_map[:]]
            ),
            x=y_linspace
        )
        
        self.volume = volume
        
    def orderString(self) -> str:
        
        return str(self.m) + str(self.n)
    
    def modeTypeString(self) -> str:
        
        return 'GH'
    
    def intensityFn(self, x_: float, y_: float):
        
        relative_x = x_ - self.center[0]
        relative_y = y_ - self.center[1]
        
        r = np.sqrt(relative_x**2 + relative_y**2)
        phi = np.arctan2(relative_y, relative_x)
        phi += self.rotation
        
        relative_x = np.cos(phi) * r
        relative_y = np.sin(phi) * r
        
        norm_coeff = 1. / (2. * self.waist**2 * np.pi * [0.5, 1, 4, 24][self.m] * [0.5, 1, 4, 24][self.n])
        
        return (1. / self.volume) * norm_coeff * (self.gh_m((2**0.5 * relative_x) / self.waist) * np.exp(-relative_x**2 / self.waist**2))**2 * (self.gh_n((2**0.5 * relative_y) / self.waist) * np.exp(-relative_y**2 / self.waist**2))**2
    
    def intensityMap(self, x_meshgrid, y_meshgrid):
        
        return self.intensityFn(x_meshgrid, y_meshgrid)
        
    
@dataclass
class Detector:
    
    waist = 1.
    
    center: np.ndarray = np.array([0.,0.])
    
    def detectFn(self, x: float, y: float, p: float):
        
        relative_x = x - self.center[0]
        relative_y = y - self.center[1]
        
        r = np.sqrt(relative_x**2 + relative_y**2)
        
        return np.exp(-(r**2 / 2) / (2 * self.waist**2)) * p

@dataclass
class Emitter:
    
    xy: np.array
    relative_brightness: float
    
@dataclass
class Solver:
    
    illumination_structures: List[Any]
    
    detector: Detector
    
    g_1_true: np.ndarray
    g_2_true: np.ndarray
    
    def __post_init__(self):
        
        self.optimization_lambda = lambda guess: self.rss(guess)
        
    def rss(self, guess):
        
        g_1_guess = np.zeros_like(self.g_1_true)
        g_2_guess = np.zeros_like(self.g_2_true)
        
        e_1_guess = Emitter(
            guess[:2],
            1.
        )
        
        e_2_guess = Emitter(
            guess[2:4],
            guess[4]
        )
        
        for is_idx in range(0,len(self.illumination_structures)):
    
            p_1_guess = e_1_guess.relative_brightness * (self.illumination_structures[is_idx].intensityFn(*e_1_guess.xy))
            p_2_guess = e_2_guess.relative_brightness * (self.illumination_structures[is_idx].intensityFn(*e_2_guess.xy))
            
            p_1_guess = self.detector.detectFn(*e_1_guess.xy, p_1_guess)
            p_2_guess = self.detector.detectFn(*e_2_guess.xy, p_2_guess)
            
            g_1_guess[is_idx] = (p_1_guess + p_2_guess) / (e_1_guess.relative_brightness + e_2_guess.relative_brightness)
            
            alpha = p_2_guess / p_1_guess
            
            g_2_guess[is_idx] = (2 * alpha) / (1 + alpha)**2
            
        return np.sum((self.g_1_true - g_1_guess)**2 + (self.g_2_true - g_2_guess)**2)
    
#     xy_emitter_1 = Emitter(ps_vec[0:2], 1.0)
#     xy_emitter_2 = Emitter(ps_vec[2:4], p_0_2)

# @dataclass
# class Detector:
    
#     xy: np.array = np.array([0.,0.])
#     w: float = 1
    
# # @dataclass
# # class GaussLaguerreTEM:
    
# #     p: int
# #     l: int
# #     rho: float

# def temModeFnXY(m: int, n: int, x: float, y: float, I_0: float = 1, w_0: float = 1, w: float = 1) -> float:
    
#     return I_0 * (w_0 / w)**2 * (physicistsHermite(m)((2**0.5 * x)/w) * np.exp(-x**2 / w**2))**2 * (physicistsHermite(n)((2**0.5 * y)/w) * np.exp(-y**2 / w**2))**2

# def fastTEM(hermite_polynomial_m: orthopoly1d, hermite_polynomial_n: orthopoly1d, x: float, y: float, I_0: float = 1, w_0: float = 1, w: float = 1) -> float:
    
#     return I_0 * (w_0 / w)**2 * (hermite_polynomial_m((2**0.5 * x)/w) * np.exp(-x**2 / w**2))**2 * (hermite_polynomial_n((2**0.5 * y)/w) * np.exp(-y**2 / w**2))**2

# def parallelTEM(m: int, n: int, y_val: float, grid_x: float, x_range) -> np.ndarray:

#     x_linspace = np.linspace(*x_range, grid_x)
    
#     intensity_vals = np.ndarray((grid_x), dtype=np.float32)
    
#     hermite_polynomial_m = physicistsHermite(m)
#     hermite_polynomial_n = physicistsHermite(n)
    
#     for x_idx in range(0, grid_x):
            
#             # intensity_vals[x_idx] = temModeFnXY(m, n, x_linspace[x_idx], y_val)
#             intensity_vals[x_idx] = fastTEM(hermite_polynomial_m, hermite_polynomial_n, x_linspace[x_idx], y_val)
            
#     return intensity_vals

# def parallelTEM_Affine(m: int, n: int, y_val: float, grid_x: float, translation: np.ndarray, rotation: np.ndarray, x_range) -> np.ndarray:

#     x_linspace = np.linspace(*x_range, grid_x)
    
#     intensity_vals = np.ndarray((grid_x), dtype=np.float32)
    
#     hermite_polynomial_m = physicistsHermite(m)
#     hermite_polynomial_n = physicistsHermite(n)
    
#     for x_idx in range(0, grid_x):
            
#             intensity_vals[x_idx] = fastTEM(hermite_polynomial_m, hermite_polynomial_n, x_linspace[x_idx] - translation[0], y_val - translation[1])
            
#     return intensity_vals

# def parallelConfocalScan(emitters: List[Emitter], light_structure: RectangularTEM, y_val, x_linspace, detector_w) -> np.ndarray:
    
#     linspace_len = np.size(x_linspace)
    
#     g_1_g_2_concatenated = np.ones(2 * linspace_len, dtype=np.float64)
    
#     for x_idx in range(0, np.size(x_linspace)):
        
#         xy_objective = np.array([x_linspace[x_idx], y_val], dtype=np.float64)
        
#         xy_emitter_1_relative = emitters[0].xy - xy_objective
#         xy_emitter_2_relative = emitters[1].xy - xy_objective
        
#         r_1 = np.linalg.norm(xy_emitter_1_relative, axis=0)
#         r_2 = np.linalg.norm(xy_emitter_2_relative, axis=0)
        
#         p_1 = emitters[0].relative_brightness * light_structure.w * fastTEM(
#             hermite_polynomial_m=light_structure.hermite_polynomial_m,
#             hermite_polynomial_n=light_structure.hermite_polynomial_n,
#             x=xy_emitter_1_relative[0],
#             y=xy_emitter_1_relative[1],
#             I_0=1,
#             w_0=1,
#             w=light_structure.w
#         )
        
#         p_2 = emitters[1].relative_brightness * light_structure.w * fastTEM(
#             hermite_polynomial_m=light_structure.hermite_polynomial_m,
#             hermite_polynomial_n=light_structure.hermite_polynomial_n,
#             x=xy_emitter_2_relative[0],
#             y=xy_emitter_2_relative[1],
#             I_0=1,
#             w_0=1,
#             w=light_structure.w
#         )
        
#         p_1 = np.exp(-(r_1**2 / 2)/(2 * detector_w**2)) * p_1
#         p_2 = np.exp(-(r_2**2 / 2)/(2 * detector_w**2)) * p_2
            
#         g_1_g_2_concatenated[x_idx] = (p_1 + p_2) / (emitters[0].relative_brightness + emitters[1].relative_brightness)
        
#         alpha = p_1 / p_2
        
#         g_1_g_2_concatenated[linspace_len + x_idx] = (2 * alpha) / (1 + alpha)**2
        
#     return g_1_g_2_concatenated

# def groundTruthG1_G2(detector: Detector, emitters: List[Emitter], light_structures: List[RectangularTEM], xy_objective):
    
#     # xy_objective = np.array([0.,0.], dtype=np.float64)
        
#     xy_emitter_1_relative = emitters[0].xy - xy_objective
#     xy_emitter_2_relative = emitters[1].xy - xy_objective
    
#     r_1 = np.linalg.norm(xy_emitter_1_relative, axis=0)
#     r_2 = np.linalg.norm(xy_emitter_2_relative, axis=0)
    
#     g_1_pred = np.ndarray((len(light_structures),1), dtype=np.float64)
#     g_2_pred = np.ndarray((len(light_structures),1), dtype=np.float64)
    
#     for light_structure_idx in range(len(light_structures)):
        
#         light_structure = light_structures[light_structure_idx]
        
#         # hermite_polynomial_m = physicistsHermite(light_structure[0][0])
#         # hermite_polynomial_n = physicistsHermite(light_structure[0][1])
        
#         p_1 = emitters[0].relative_brightness * light_structure.norm_coeff * fastTEM(
#             hermite_polynomial_m=light_structure.hermite_polynomial_m,
#             hermite_polynomial_n=light_structure.hermite_polynomial_n,
#             x=xy_emitter_1_relative[0],
#             y=xy_emitter_1_relative[1],
#             I_0=1,
#             w_0=1,
#             w=light_structure.w
#         )
        
#         p_2 = emitters[1].relative_brightness * light_structure.norm_coeff * fastTEM(
#             hermite_polynomial_m=light_structure.hermite_polynomial_m,
#             hermite_polynomial_n=light_structure.hermite_polynomial_n,
#             x=xy_emitter_2_relative[0],
#             y=xy_emitter_2_relative[1],
#             I_0=1,
#             w_0=1,
#             w=light_structure.w
#         )
        
#         p_1 = np.exp(-(r_1**2 / 2)/(2 * detector.w**2)) * p_1
#         p_2 = np.exp(-(r_2**2 / 2)/(2 * detector.w**2)) * p_2
            
#         g_1_pred[light_structure_idx] = (p_1 + p_2) / (emitters[0].relative_brightness + emitters[1].relative_brightness)
        
#         alpha = p_2 / p_1
        
#         g_2_pred[light_structure_idx] = (2 * alpha) / (1 + alpha)**2
        
#     return g_1_pred,g_2_pred

# def optimizeMe(ps_vec, detector: Detector, emitters: List[Emitter], light_structures: List[RectangularTEM], xy_objective, noisy_g_1, noisy_g_2):
    
#     p_0_2 = ps_vec[4]
    
#     xy_emitter_1 = Emitter(ps_vec[0:2], 1.0)
#     xy_emitter_2 = Emitter(ps_vec[2:4], p_0_2)
    
#     g_1_pred = np.ndarray((len(light_structures),1), dtype=np.float64)
#     g_2_pred = np.ndarray((len(light_structures),1), dtype=np.float64)
    
#     for light_structure_idx in range(len(light_structures)):
        
#         light_structure = light_structures[light_structure_idx]
        
#         # hermite_polynomial_m = physicistsHermite(light_structure[0][0])
#         # hermite_polynomial_n = physicistsHermite(light_structure[0][1])
        
#         xy_emitter_1_relative = xy_emitter_1.xy - xy_objective
#         xy_emitter_2_relative = xy_emitter_2.xy - xy_objective
        
#         r_1 = np.linalg.norm(xy_emitter_1_relative, axis=0)
#         r_2 = np.linalg.norm(xy_emitter_2_relative, axis=0)
        
#         p_1 = emitters[0].relative_brightness * light_structure.norm_coeff * fastTEM(
#             hermite_polynomial_m=light_structure.hermite_polynomial_m,
#             hermite_polynomial_n=light_structure.hermite_polynomial_n,
#             x=xy_emitter_1_relative[0],
#             y=xy_emitter_1_relative[1],
#             I_0=1,
#             w_0=1,
#             w=light_structure.w
#         )
        
#         p_2 = emitters[1].relative_brightness * light_structure.norm_coeff * fastTEM(
#             hermite_polynomial_m=light_structure.hermite_polynomial_m,
#             hermite_polynomial_n=light_structure.hermite_polynomial_n,
#             x=xy_emitter_2_relative[0],
#             y=xy_emitter_2_relative[1],
#             I_0=1,
#             w_0=1,
#             w=light_structure.w
#         )
        
#         p_1 = np.exp(-(r_1**2 / 2)/(2 * detector.w**2)) * p_1
#         p_2 = np.exp(-(r_2**2 / 2)/(2 * detector.w**2)) * p_2
        
#         g_1_pred[light_structure_idx] = (p_1 + p_2) / (1. + p_0_2)
        
#         alpha = p_2 / p_1
                
#         g_2_pred[light_structure_idx] = (2 * alpha) / (1 + alpha)**2
        
#     return np.sum((g_1_pred - noisy_g_1)**2) + np.sum((g_2_pred - noisy_g_2)**2)