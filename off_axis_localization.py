import numpy as np
from parula import parula
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import KDTree

matplotlib.rcParams['text.usetex'] = True

from qclm import Emitter, GaussLaguerre, GaussHermite, Detector, Solver, QuadTree, Rect, Point
from scipy.optimize import minimize
from scipy.optimize import fmin

import os

path = os.path.dirname(__file__)

grid_x = 3750
grid_y = 3750
waists = 0.55

def main() -> None:
    
    # our detector 
    
    detector = Detector()
    
    x_range = (-detector.waist / 2, detector.waist / 2)
    y_range = (-detector.waist / 2, detector.waist / 2)
    
    x_linspace = np.linspace(*x_range, grid_x)
    y_linspace = np.linspace(*y_range, grid_y)
    
    # light structures
    
    illumination_structures = [
        GaussHermite(0, 0, 1., 0.5, center=[0.0,0.0], rotation=0),
        GaussHermite(1, 0, 1., 0.5, center=[0.0,0.0], rotation=0*np.pi/10),
        GaussHermite(1, 0, 1., 0.5, center=[0.0,0.0], rotation=1*np.pi/10),
        GaussHermite(1, 0, 1., 0.5, center=[0.0,0.0], rotation=2*np.pi/10),
        GaussHermite(1, 0, 1., 0.5, center=[0.0,0.0], rotation=3*np.pi/10),
        GaussHermite(1, 0, 1., 0.5, center=[0.0,0.0], rotation=4*np.pi/10),
        GaussHermite(1, 0, 1., 0.5, center=[0.0,0.0], rotation=5*np.pi/10)
    ]
    
    # emitters
    
    e_1 = Emitter(
        np.array([-0.3,0.2]),
        1.0
    )

    e_2 = Emitter(
        np.array([0.2,0.1]),
        0.5
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
        
    g_1_true = np.array(g_1_true) * (1 + 0.05 * np.random.randn(len(illumination_structures),1))
    g_2_true = np.array(g_2_true) * (1 + 0.05 * np.random.randn(len(illumination_structures),1))
    # g_2_true = np.min(g_2_true, 0.5)
    g_2_true = np.array([min(g_2,np.array(0.5)) for g_2 in g_2_true])
        
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
    
        x_0 = 0.25 * np.random.randn(5,1)
        x_0[4] = 0.5
        
        x_opt[trial_idx], _, _, _, _ = fmin(func=solver.optimization_lambda, x0=x_0, disp=False, full_output=True)
        
    print(np.mean(x_opt, axis=0))
    
    all_points = np.ndarray((trials * 2,2))
    
    all_points[:trials,:] = x_opt[:,:2]
    all_points[trials:,:] = x_opt[:,2:4]
    
    total_points_e_1 = []
    total_points_e_2 = []
    
    e_1_xy_distances = np.linalg.norm(x_opt[:,:2][:]-e_1.xy.T, axis=1)
    e_2_xy_distances = np.linalg.norm(x_opt[:,2:4][:]-e_2.xy.T, axis=1)
    
    total_points_e_1 += list(np.where(e_1_xy_distances < 0.05)[0])
    total_points_e_2 += list(np.where(e_2_xy_distances < 0.05)[0])
    
    e_1_xy_distances = np.linalg.norm(x_opt[:,2:4][:]-e_2.xy.T, axis=1)
    e_2_xy_distances = np.linalg.norm(x_opt[:,:2][:]-e_1.xy.T, axis=1)
    
    total_points_e_1 += list(np.where(e_1_xy_distances < 0.05)[0])
    total_points_e_2 += list(np.where(e_2_xy_distances < 0.05)[0])
    
    print(len(total_points_e_1) / trials)
    print(len(total_points_e_2) / trials)
    
    np.savetxt(os.path.join(path,'localization-results','positions.csv'), x_opt, delimiter=',')
    
    plt.scatter(x_opt[0:,0],x_opt[0:,1], c='b', s=2., marker='.')
    plt.scatter(x_opt[0:,2],x_opt[0:,3], c='r', s=2., marker='.')
    plt.scatter(e_1.xy[0], e_1.xy[1], c='r', marker='x', s=40, linewidths=1)
    plt.scatter(e_2.xy[0], e_2.xy[1], facecolors='none', edgecolors='r', marker='o', s=20, linewidths=1)
    plt.xlim(-waists * detector.waist,waists * detector.waist)
    plt.ylim(-waists * detector.waist,waists * detector.waist)
    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel(r"$y$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(f'localization-results/localization.png', dpi=400, bbox_inches='tight')
    plt.close()
    
    points = [Point(*coord) for coord in all_points[:]]
    
    domain = Rect(0, 0, 2 * waists * detector.waist, 2 * waists * detector.waist)
    qtree = QuadTree(domain, int(0.05 * trials))
    for point in points:
        qtree.insert(point)

    ax = plt.subplot()
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    qtree.draw(ax)

    plt.scatter(all_points[:,0],all_points[:,1], color='cyan', s=2., marker='.')
    plt.scatter(e_1.xy[0], e_1.xy[1], c='r', marker='x', s=40, linewidths=1)
    plt.scatter(e_2.xy[0], e_2.xy[1], facecolors='none', edgecolors='r', marker='o', s=20, linewidths=1)
    plt.xlim(-waists * detector.waist,waists * detector.waist)
    plt.ylim(-waists * detector.waist,waists * detector.waist)
    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel(r"$y$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(os.path.join(path,'localization-results','localization_quadtree.png'), dpi=600, bbox_inches='tight')
    # plt.show()
    
    
    
if __name__ == '__main__':
    
    main()