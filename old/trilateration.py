from scipy.special import hermite as physicistsHermite
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing as mp
from tem_utils import *
from parula import parula
import scipy.optimize
import random

def sumSquareDifferences(ps_vec, noisy_g_1, noisy_g_2, detector_w, emitters, light_structures, xy_objective) -> float:
    
    g_1_pred = np.ndarray((len(light_structures),1), dtype=np.float64)
    g_2_pred = np.ndarray((len(light_structures),1), dtype=np.float64)
    
    p_0_2 = ps_vec[4]
    
    xy_emitter_1 = ps_vec[0:2]
    xy_emitter_2 = ps_vec[2:4]
    
    for light_structure_idx in range(len(light_structures)):
        
        light_structure = light_structures[light_structure_idx]
        
        hermite_polynomial_m = physicistsHermite(light_structure[0][0])
        hermite_polynomial_n = physicistsHermite(light_structure[0][1])
        
        xy_emitter_1_relative = xy_emitter_1 - xy_objective
        xy_emitter_2_relative = xy_emitter_2 - xy_objective
        
        r_1 = np.linalg.norm(xy_emitter_1_relative, axis=0)
        r_2 = np.linalg.norm(xy_emitter_2_relative, axis=0)
        
        p_1 = 1. * light_structure[-1] * fastTEM(
            hermite_polynomial_m=hermite_polynomial_m,
            hermite_polynomial_n=hermite_polynomial_n,
            x=xy_emitter_1_relative[0],
            y=xy_emitter_1_relative[1],
            I_0=1,
            w_0=1,
            w=light_structure[1]
        )
        
        p_2 = p_0_2 * light_structure[-1] * fastTEM(
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
        
        g_1_pred[light_structure_idx] = (p_1 + p_2) / (1. + p_0_2)
        
        alpha = p_2 / p_1
                
        g_2_pred[light_structure_idx] = (2 * alpha) / (1 + alpha)**2
        
    return np.sum((g_1_pred - noisy_g_1)**2) + np.sum((g_2_pred - noisy_g_2)**2)

