import numpy as np
from parula import parula
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import KDTree
# from mayavi import mlab
# mlab.options.offscreen = True
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
from scipy.spatial.transform import Rotation
import os

matplotlib.rcParams['text.usetex'] = True

# from qclm import Emitter, GaussLaguerre, GaussHermite, Detector, Solver, QuadTree, Rect, Point, genHermite, mlab_imshowColor
from scipy.optimize import minimize
from scipy.optimize import fmin

path = os.path.dirname(__file__)

#==================================================================================================
# Configuration - edit me!
#==================================================================================================

CORE_OVERRIDE = True

TRIALS = 200

ROWS = 4
COLS = 4

CORE_PSF = 1.

# G2_CAPABLE_IDX = [ 5,9,10 ]

X_DIFF = 1.25

cores: np.ndarray

if CORE_OVERRIDE:
    
    cores = np.array([
        [0,1,0],
        [-np.sin(60*np.pi/180),-np.cos(60*np.pi/180),0],
        [np.sin(60*np.pi/180),-np.cos(60*np.pi/180),0]
    ], dtype=np.float64)
    
    G2_CAPABLE_IDX = [ 0,1,2 ]

EMITTER_XY = 0.5 * np.array([
    [-0.6300,-0.1276,0],
    [0.5146,-0.5573,0]
], dtype=np.float64)

EMITTER_BRIGHTNESS = np.array([
    1.,
    0.3617
], dtype=np.float64)

#==================================================================================================
# Deduced configuration values - don't touch this!
#==================================================================================================

ROW_SHIFT = X_DIFF / 2

R_VERTEX = ROW_SHIFT * 2 / 3**0.5

Y_DIFF = 1.5 * R_VERTEX

col_indexes = np.arange(COLS)
row_indexes = np.arange(ROWS)

if not CORE_OVERRIDE:

    cores = np.ndarray((ROWS * COLS,3), dtype=np.float64)

    cores[:,2] = 0

    for row in range(ROWS):
        
        for col in range(COLS):
            
            cores[row * COLS:(row + 1) * COLS,0] = range(COLS)
            cores[row * COLS:(row + 1) * COLS,1] = row

    cores[:,0] *= X_DIFF
    cores[:,0] += np.mod(cores[:,1], 2) * ROW_SHIFT
    cores[:,0] -= (ROW_SHIFT + X_DIFF * (COLS - 1)) / 2
    cores[:,1] *= Y_DIFF
    cores[:,1] -= (Y_DIFF * (ROWS - 1)) / 2
    
G1_ONLY_IDX = [idx for idx in range(np.shape(cores)[0]) if idx not in G2_CAPABLE_IDX]

def main():

    #==================================================================================================
    # Localization process
    #==================================================================================================

    emitter_distances = np.ndarray((np.shape(cores)[0], np.shape(EMITTER_XY)[0]), dtype=np.float64)

    for emitter_idx in range(np.shape(EMITTER_XY)[0]):
        
        diffv = cores - EMITTER_XY[emitter_idx,:]
        
        emitter_distances[:,emitter_idx] = np.linalg.norm(diffv, axis=1)
        
    powers = np.exp(-(emitter_distances**2/2)/(2*CORE_PSF**2))

    for emitter_idx in range(np.shape(EMITTER_XY)[0]):
        
        powers[:,emitter_idx] *= EMITTER_BRIGHTNESS[emitter_idx]
        
    g1_true = (powers[:,0] + powers[:,1]) / (EMITTER_BRIGHTNESS[0] + EMITTER_BRIGHTNESS[1])

    alpha = powers[G2_CAPABLE_IDX,0] / powers[G2_CAPABLE_IDX,1]

    g2_true = (2*alpha)/((1+alpha)**2)
    
    x_opt = np.ndarray((TRIALS, 5), dtype=np.float64)
    
    for trial_idx in range(TRIALS):
        
        print(trial_idx)
        
        g1_noisy = (g1_true * np.random.uniform(low=(1-0.1),high=(1+0.1), size=(np.shape(g1_true)[0],1)))
        g2_noisy = (g2_true * np.random.uniform(low=(1-0.1),high=(1+0.1), size=(np.shape(g2_true)[0],1)))
        
        x_0 = np.random.randn(5,1)
        x_0[4] = 0.5
        
        g1_mat_measure = np.array([0.783340092207118,
0.818150251986132,
0.693069161471851], dtype=np.float64)
    
        g2_mat_measure = np.array([0.364697000122009,
0.303173696912229,
0.407550413831280], dtype=np.float64)
        
        xx0 = np.array([0.0583901331581728,0.485090888610420,0.128934147111779,0.672737850669774,0.500000000000000], dtype=np.float64)
            
        rss_lambda = lambda guess_x: rssFn(guess_x, g1_noisy, g2_noisy)
        
        # x_opt[trial_idx,:], _, iter, _, _ = scipy.optimize.fmin(func=rss_lambda, x0=x_0, disp=False, full_output=True, maxiter=400)
        
        result = scipy.optimize.minimize(fun=rss_lambda, x0=x_0, method='Nelder-Mead', tol=1e-8)
            
        x_opt[trial_idx,:] = result.x
        
        if x_opt[trial_idx,4] > 1:
            
            x_opt[trial_idx,4] = 1. / x_opt[trial_idx,4]
            
            tmp = x_opt[trial_idx,0:2]
            
            x_opt[trial_idx,0:2] = x_opt[trial_idx,2:4]
            
            x_opt[trial_idx,2:4] = tmp
            
    print(np.mean(x_opt, axis=0))
    
    all_points = np.ndarray((TRIALS * 2,2))
    
    all_points[:TRIALS,:] = x_opt[:,:2]
    all_points[TRIALS:,:] = x_opt[:,2:4]
    
    plt.scatter(x_opt[:,0], x_opt[:,1], color='cyan', s=2., marker='.')
    plt.scatter(all_points[TRIALS:,0],all_points[TRIALS:,1], color='magenta', s=2., marker='.')
    plt.scatter(cores[G1_ONLY_IDX,0], cores[G1_ONLY_IDX,1], c='b', marker='o')
    plt.scatter(cores[G2_CAPABLE_IDX,0], cores[G2_CAPABLE_IDX,1], c='b', marker='x')
    plt.scatter(EMITTER_XY[:,0], EMITTER_XY[:,1], c='r', marker='+')
    plt.xlim(-2.5,2.5)
    plt.ylim(-2.5,2.5)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(os.path.join(path,'bad_localization.png'), dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()

def rssFn(x_guess, g1_noisy, g2_noisy):
    
    emitter_xy_guess = np.array([
        [*x_guess[0:2],0],
        [*x_guess[2:4],0]
    ], dtype=np.float64)

    emitter_brightness_guess = np.array([
        1.,
        x_guess[4]
    ], dtype=np.float64)
    
    emitter_distances = np.ndarray((np.shape(cores)[0], np.shape(emitter_xy_guess)[0]), dtype=np.float64)

    for emitter_idx in range(np.shape(emitter_xy_guess)[0]):
        
        emitter_distances[:,emitter_idx] = np.linalg.norm(cores - emitter_xy_guess[emitter_idx,:], axis=1)
        
    powers = np.exp(-(emitter_distances**2/2)/(2*CORE_PSF**2))

    for emitter_idx in range(np.shape(emitter_xy_guess)[0]):
        
        powers[:,emitter_idx] *= emitter_brightness_guess[emitter_idx]
        
    g1_guess = (powers[:,0] + powers[:,1]) / (emitter_brightness_guess[0] + emitter_brightness_guess[1])

    alpha = powers[G2_CAPABLE_IDX,0] / powers[G2_CAPABLE_IDX,1]

    g2_guess = (2*alpha)/((1+alpha)**2)
    
    g1_diffs = g1_guess - g1_noisy
    g2_diffs = g2_guess - g2_noisy
    
    return np.sum(g1_diffs**2) + np.sum(g2_diffs**2)
    
if __name__ == '__main__':
    
    main()