import numpy as np
from parula import parula
import matplotlib.pyplot as plt
import matplotlib
from qclm import Emitter, GaussLaguerre, GaussHermite, Detector, Solver
from scipy.optimize import minimize

def main() -> None:
    
    # our detector 
    
    detector = Detector()
    
    # light structures
    
    illumination_structures = [
        GaussHermite(3, 1, 1., 0.25, center=0.2 * np.random.randn(2,1), rotation=np.pi * np.random.randn()),
        GaussHermite(0, 0, 1., 0.25, center=0.2 * np.random.randn(2,1), rotation=np.pi * np.random.randn()),
        GaussHermite(2, 1, 1., 0.25, center=0.2 * np.random.randn(2,1), rotation=np.pi * np.random.randn()),
        GaussLaguerre(1, 3, 1., 0.25, center=0.2 * np.random.randn(2,1), rotation=np.pi * np.random.randn()),
        GaussLaguerre(0, 3, 1., 0.25, center=0.2 * np.random.randn(2,1), rotation=np.pi * np.random.randn()),
        GaussLaguerre(1, 2, 1., 0.25, center=0.2 * np.random.randn(2,1), rotation=np.pi * np.random.randn())
    ]
    
    # emitters
    
    e_1 = Emitter(
        np.array([-0.3,0.2]),
        1.0
    )

    e_2 = Emitter(
        np.array([0.2,0.1]),
        0.7
    )
    
    g_1_true = np.ndarray((len(illumination_structures),1))
    g_2_true = np.ndarray((len(illumination_structures),1))
    
    for is_idx in range(0,len(illumination_structures)):
    
        p_1_true = e_1.relative_brightness * (illumination_structures[is_idx].intensityFn(*e_1.xy))
        p_2_true = e_2.relative_brightness * (illumination_structures[is_idx].intensityFn(*e_2.xy))
        
        p_1_true = detector.detectFn(*e_1.xy, p_1_true)
        p_2_true = detector.detectFn(*e_2.xy, p_2_true)
        
        g_1_true[is_idx] = (p_1_true + p_2_true) / (e_1.relative_brightness + e_2.relative_brightness)
        
        alpha = p_2_true / p_1_true
        
        g_2_true[is_idx] = (2 * alpha) / (1 + alpha)**2
        
    solver = Solver(
        illumination_structures,
        detector,
        g_1_true,
        g_2_true
    )
    
    trials = 200
    
    x_opt = np.ndarray((trials,5))
    
    for trial_idx in range(trials):
        
        print(trial_idx)
        
        # print(trial_idx)
    
        x_0 = 0.25 * np.random.randn(5,1)
        x_0[4] = 0.5
        
        opt_result = minimize(
            fun=solver.optimization_lambda,
            x0=x_0,
            method='Nelder-Mead',
            bounds=[(-1,1),(-1,1),(-1,1),(-1,1),(0,1)]
        )
        
        x_opt[trial_idx,:] = opt_result.x
        
    print(np.mean(x_opt, axis=0))
    
    plt.scatter(x_opt[0:,0],x_opt[0:,1], c='b', s=2., marker='.')
    plt.scatter(x_opt[0:,2],x_opt[0:,3], c='r', s=2., marker='.')
    # plt.scatter(xy_objective[0], xy_objective[1], c='r', marker='.', s=40, linewidths=1.0, label=r'$\mathrm{Objective\;Location}$')
    # plt.scatter(max_x,max_y, c='r', marker='x', s=40, linewidths=1.0, label=r'$\mathrm{Max\;Intensity}$')
    # plt.scatter(emitters[0].xy[0],emitters[0].xy[1], c='k', marker='x', s=40, linewidths=1.0)
    # plt.scatter(emitters[1].xy[0],emitters[1].xy[1], c='k', marker='x', s=40, linewidths=1.0)
    # plt.scatter(x_opt[1,0],x_opt[1,1], c='k', s=40, marker='+', linewidths=0.5)
    # plt.scatter(x_opt[1,2],x_opt[1,3], c='k', s=40, marker='+', linewidths=0.5)    
    # plt.xlim(*x_range)
    # plt.ylim(*y_range)
    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel(r"$y$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.show()
    
    
    
if __name__ == '__main__':
    
    main()