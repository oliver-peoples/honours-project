import numpy as np
from parula import parula
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

from qclm import Emitter, GaussLaguerre, GaussHermite, Detector, Solver
from scipy.optimize import minimize
from scipy.optimize import fmin

scan_confocally = True

def main() -> None:
    
    # the three detector form
    
    uniform_illumination = GaussHermite(0, 0, 1., 1000)
    
    detectors = [
        Detector([0,1]),
        Detector([-np.sin(60*np.pi/180),-np.cos(60*np.pi/180)]),
        Detector([np.sin(60*np.pi/180),-np.cos(60*np.pi/180)])
    ]
    
    e_1 = Emitter(
        np.array([-0.6300,-0.1276]),
        1.0
    )

    e_2 = Emitter(
        np.array([0.5146,0.5573]),
        0.3617
    )
    
    # do a confocal scan
    
    # if scan_confocally:
        
        
    
    g_1_true = np.ndarray((3,1))
    g_2_true = np.ndarray((3,1))
    
    for detector_idx in range(len(detectors)):
        
        p_1 = e_1.relative_brightness * uniform_illumination.intensityFn(*e_1.xy)
        p_2 = e_2.relative_brightness * uniform_illumination.intensityFn(*e_2.xy)
        
        p_1 = detectors[detector_idx].detectFn(*e_1.xy, p_1)
        p_2 = detectors[detector_idx].detectFn(*e_2.xy, p_2)
        
        alpha = p_2 / p_1
        
        g_1_true[detector_idx] = (p_1 + p_2) / (e_1.relative_brightness + e_2.relative_brightness)
        g_2_true[detector_idx] = (2 * alpha) / (1 + alpha)**2
        
    print(g_1_true, g_2_true)
    
    def rss(x):
        
        g_1_guess = np.ndarray((3,1))
        g_2_guess = np.ndarray((3,1))
        
        e_1_guess = Emitter(xy=x[:2], relative_brightness=1.)
        e_2_guess = Emitter(xy=x[2:4], relative_brightness=x[4])
        
        for detector_idx in range(len(detectors)):
            
            p_1 = e_1_guess.relative_brightness * uniform_illumination.intensityFn(*e_1.xy)
            p_2 = e_2_guess.relative_brightness * uniform_illumination.intensityFn(*e_2.xy)
            
            p_1 = detectors[detector_idx].detectFn(*e_1.xy, p_1)
            p_2 = detectors[detector_idx].detectFn(*e_2.xy, p_2)
            
            alpha = p_2 / p_1
            
            g_1_guess[detector_idx] = (p_1 + p_2) / (e_1_guess.relative_brightness + e_2_guess.relative_brightness)
            g_2_guess[detector_idx] = (2 * alpha) / (1 + alpha)**2
            
        return np.sum((g_1_true - g_1_guess)**2) + np.sum((g_2_true - g_2_guess)**2)
    
    optimization_lambda = lambda guess: rss(guess)
    
    trials = 200
    
    x_opt = np.ndarray((trials,5))
    
    for trial_idx in range(trials):
        
        print(trial_idx)
        
        # print(trial_idx)
    
        x_0 = 0.25 * np.random.randn(5,1)
        x_0[4] = 0.5
        
        # x_0 = np.array([*e_1.xy,*e_2.xy,e_2.relative_brightness])
        
        # opt_result = minimize(
        #     fun=optimization_lambda,
        #     x0=x_0,
        #     method='Nelder-Mead',
        #     bounds=[(-1,1),(-1,1),(-1,1),(-1,1),(0,1)]
        # )
        
        # x_opt[trial_idx,:] = opt_result.x
        
        x_opt[trial_idx], _, _, _, _ = fmin(func=optimization_lambda, x0=x_0, disp=False, full_output=True)
        
    print(np.mean(x_opt, axis=0))
    
    plt.scatter(x_opt[0:,0],x_opt[0:,1], c='b', s=2., marker='.')
    plt.scatter(x_opt[0:,2],x_opt[0:,3], c='r', s=2., marker='.')
    plt.scatter(e_1.xy[0], e_1.xy[1], c='r', marker='x', s=40, linewidths=1)
    plt.scatter(e_2.xy[0], e_2.xy[1], facecolors='none', edgecolors='r', marker='o', s=20, linewidths=1)
    plt.scatter(detectors[0].center[0], detectors[0].center[1], c='k', marker='x', s=20, linewidths=1)
    plt.scatter(detectors[1].center[0], detectors[1].center[1], c='k', marker='x', s=20, linewidths=1)
    plt.scatter(detectors[2].center[0], detectors[2].center[1], c='k', marker='x', s=20, linewidths=1)
    # plt.xlim(-detector.waist/2,detector.waist/2)
    # plt.ylim(-detector.waist/2,detector.waist/2)
    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel(r"$y$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'localization-results/localization.png', dpi=400, bbox_inches='tight')
    
if __name__ == '__main__':
    
    main()

# %x1 = [-0.5150,    0.8734];
# %x2 = [0.7204,   -0.2055];

# %Strange set 2
# %x1 = [-0.0470,   -0.6462];
# %x2 = [0.0058    0.4619];

# %Position of detectors
# x0 = 1.0*[0,1;
#     -sin(60*pi/180),-cos(60*pi/180);
#     sin(60*pi/180),-cos(60*pi/180)];

# sigma = 1; %sigma = 0.21 * \ambda/NA - standard deviation of PSF
# %Powers
# P01 = 1;
# %P02 = 0.5;
# %P02 = 0.8;

# %Strange set
# %P02 = 0.4794;
# %Strange set 2
# %P02 = 0.8819;

# %Data for figure
# P02 = 0.3617;
# x1 = [-0.6300,   -0.1276];
# x2 = [0.5146,   -0.5573];