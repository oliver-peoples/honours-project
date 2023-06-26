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

matplotlib.rcParams['text.usetex'] = True

def sumSquareDifferences(ps_vec, noisy_g_1, noisy_g_2, detector_w, emitters, light_structures) -> float:
    
    xy_objective = np.array([0., 0.], dtype=np.float64)
    
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

def groundTruthG1_G2(detector_w, emitters, light_structures):
    
    xy_objective = np.array([0.,0.], dtype=np.float64)
        
    xy_emitter_1_relative = emitters[0][0] - xy_objective
    xy_emitter_2_relative = emitters[1][0] - xy_objective
    
    r_1 = np.linalg.norm(xy_emitter_1_relative, axis=0)
    r_2 = np.linalg.norm(xy_emitter_2_relative, axis=0)
    
    g_1_pred = np.ndarray((len(light_structures),1), dtype=np.float64)
    g_2_pred = np.ndarray((len(light_structures),1), dtype=np.float64)
    
    for light_structure_idx in range(len(light_structures)):
        
        light_structure = light_structures[light_structure_idx]
        
        hermite_polynomial_m = physicistsHermite(light_structure[0][0])
        hermite_polynomial_n = physicistsHermite(light_structure[0][1])
        
        p_1 = emitters[0][1] * light_structure[-1] * fastTEM(
            hermite_polynomial_m=hermite_polynomial_m,
            hermite_polynomial_n=hermite_polynomial_n,
            x=xy_emitter_1_relative[0],
            y=xy_emitter_1_relative[1],
            I_0=1,
            w_0=1,
            w=light_structure[1]
        )
        
        p_2 = emitters[1][1] * light_structure[-1] * fastTEM(
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
            
        g_1_pred[light_structure_idx] = (p_1 + p_2) / (emitters[0][1] + emitters[1][1])
        
        alpha = p_2 / p_1
        
        g_2_pred[light_structure_idx] = (2 * alpha) / (1 + alpha)**2
        
    return g_1_pred,g_2_pred

def main() -> None:
    
    grid_x = 750
    grid_y = 750
    
    iteration_limit = 200

    pm_thresh = 0.1 # emitter start location noise
    noise_thresh = 0.05
    range_modifier = 0.3
    
    # detector standard deviation
    
    detector_w = 1.    # declare the light structures
    
    emitters = [
        [np.array([-0.6300,-0.1276]) * range_modifier, 1.0000],
        [np.array([0.51460,-0.5573]) * range_modifier, 0.3167]
    ]
    
    print(f'Emitter separation: {np.linalg.norm(emitters[0][0]-emitters[1][0])}')
    
    # extract just the positions
    
    emitter_positions = np.stack([emitter[0] for emitter in emitters])
    
    emitter_count = len(emitters)
    
    # obtain plot center
    
    plot_center = np.array([0., 0.])
    
    for emitter in emitters:
        
        plot_center += emitter[0]
        
    plot_center *= 1. / emitter_count
    
    light_structures = [
        [[0,0], 1., 0.],
        [[1,0], 1., 0.],
        [[0,1], 1., 0.],
        # [[1,1], 1., 0.],
        # [[2,1], 1., 0.],
        # [[1,2], 1., 0.],
        # [[2,2], 1., 0.]
    ]
    
    # initialize their normalization factors
    
    for light_structure in light_structures:
        
        light_structure.append(normalizationCoefficient(light_structure[0][0], light_structure[0][1], light_structure[1]))
        
    # get truth g_1 and g_2 values
    
    truth_g_1,truth_g_2 = groundTruthG1_G2(detector_w=detector_w, emitters=emitters, light_structures=light_structures)
    
    recordings = 10
    
    xopt = np.zeros(shape=(recordings,5))
    
    # x0 = np.random.uniform(low=(-1.),high=(1.), size=(5,1)).flatten()
    # x0[4] = 0.5
    
    # for recording_idx in range(recordings):
        
    #     iter = iteration_limit + 1
        
    #     noisy_g_1 = (truth_g_1 * np.random.uniform(low=(1-noise_thresh),high=(1+noise_thresh), size=(len(light_structures),1)))
    #     noisy_g_2 = (truth_g_2 * np.random.uniform(low=(1-noise_thresh),high=(1+noise_thresh), size=(len(light_structures),1)))
        
    #     optimization_lambda = lambda ps_vec: sumSquareDifferences(ps_vec, noisy_g_1, noisy_g_2, detector_w, emitters, light_structures)
        
    #     while iter > iteration_limit:
            
    #         og_alpha = x0[4]
        
    #         xopt[recording_idx,:], _, iter, _, _ = scipy.optimize.fmin(func=optimization_lambda, x0=x0, disp=False, full_output=True, maxiter=200)
            
    #         if xopt[recording_idx,4] < 0. or xopt[recording_idx,4] > 1.:
                
    #             print(x0)
                
    #             x0[4] = og_alpha
    #             iter = iteration_limit + 1
    #             continue
    #         else:
    #             xopt[recording_idx,:4] += pm_thresh * np.random.uniform(low=(-1.),high=(1.), size=(4,1)).flatten()
          
    #     print(xopt[recording_idx])  
    #     print(f'recording {recording_idx} solved')

    #     print(optimization_lambda(x0))
        
    # avg_xopt = np.mean(xopt, 0)

    # plt.scatter(xopt[:,0],xopt[:,1], c='b', s=1., marker='.')
    # plt.scatter(xopt[:,2],xopt[:,3], c='r', s=1., marker='.')
    # plt.scatter(emitters[0][0][0],emitters[0][0][1], c='k', marker='+', s=20, linewidths=1.0)
    # plt.scatter(emitters[1][0][0],emitters[1][0][1], c='k', marker='+', s=20, linewidths=1.0)
    # plt.scatter(avg_xopt[0],avg_xopt[1], c='k', s=20, marker='x', linewidths=0.5)
    # plt.scatter(avg_xopt[2],avg_xopt[3], c='k', s=20, marker='x', linewidths=0.5)
    
    # x_range = (-0.75 * detector_w + plot_center[0], 0.75 * detector_w + plot_center[0])
    # y_range = (-0.75 * detector_w + plot_center[1], 0.75 * detector_w + plot_center[1])
    
    # plt.xlim(*x_range)
    # plt.ylim(*y_range)
    # plt.xlabel(r"$x/w$", fontsize=18)
    # plt.ylabel(r"$y/w$", fontsize=18)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.gca().set_aspect(1)
    # plt.tight_layout()
    # plt.savefig('xopt.png', dpi=600, bbox_inches='tight')
    
    for light_structure_idx in range(len(light_structures)):
        
        light_structure = light_structures[light_structure_idx]
        
        x_range = (-1.5 * detector_w + plot_center[0], 1.5 * detector_w + plot_center[0])
        y_range = (-1.5 * detector_w + plot_center[1], 1.5 * detector_w + plot_center[1])
        
        y_linspace = np.linspace(*y_range, grid_y)
        x_linspace = np.linspace(*x_range, grid_x)
        
        arg_vec = [(emitters, light_structure, y_val, x_linspace, detector_w) for y_val in y_linspace]
        
        multiprocessing_pool = mp.Pool(processes=mp.cpu_count())
        g_1_g_2_concatenated = np.stack(multiprocessing_pool.starmap(parallelConfocalScan, arg_vec), axis=0)
        multiprocessing_pool.close()
        
        g_1_scan = g_1_g_2_concatenated[:,0:grid_x]
        g_2_scan = g_1_g_2_concatenated[:,grid_x:2*grid_x]
        
        fig, ax = plt.subplots(2, 1, figsize=(8, 8))
        
        plt.subplot(2,1,1)
        
        exec('plt.title(r"$\mathbf{TEM}_{' + str(light_structure[0][0]) + ',' + str(light_structure[0][1]) + '}$", fontsize=22)')
        heatmap = plt.pcolormesh(x_linspace, y_linspace, g_1_scan, cmap=parula)
        # plt.scatter(xopt[:,0],xopt[:,1], c='b', s=1., marker='.')
        # plt.scatter(xopt[:,2],xopt[:,3], c='r', s=1., marker='.')
        plt.scatter(emitters[0][0][0],emitters[0][0][1], c='k', marker='+', s=20, linewidths=1.0)
        plt.scatter(emitters[1][0][0],emitters[1][0][1], c='k', marker='+', s=20, linewidths=1.0)
        # plt.scatter(avg_xopt[0],avg_xopt[1], c='k', s=20, marker='x', linewidths=0.5)
        # plt.scatter(avg_xopt[2],avg_xopt[3], c='k', s=20, marker='x', linewidths=0.5)
        plt.xlim(*x_range)
        plt.ylim(*y_range)
        plt.gca().set_xticklabels([])
        plt.ylabel(r"$y/w$", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        cbar = plt.colorbar(heatmap, pad=0.01)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(r'$g_{2}^{(1)}$', fontsize=18, rotation=0, labelpad=15)
        plt.gca().set_aspect(1)
        plt.tight_layout()
        
        # plt g_2
        
        plt.subplot(2,1,2)            
        heatmap = plt.pcolormesh(x_linspace, y_linspace, g_2_scan, cmap=parula)
        # plt.scatter(xopt[:,0],xopt[:,1], c='b', s=1., marker='.')
        # plt.scatter(xopt[:,2],xopt[:,3], c='r', s=1., marker='.')
        plt.scatter(emitters[0][0][0],emitters[0][0][1], c='k', marker='+', s=20, linewidths=1.0)
        plt.scatter(emitters[1][0][0],emitters[1][0][1], c='k', marker='+', s=20, linewidths=1.0)
        # plt.scatter(avg_xopt[0],avg_xopt[1], c='k', s=20, marker='x', linewidths=0.5)
        # plt.scatter(avg_xopt[2],avg_xopt[3], c='k', s=20, marker='x', linewidths=0.5)
        plt.xlim(*x_range)
        plt.ylim(*y_range)
        plt.xlabel(r"$x/w$", fontsize=18)
        plt.ylabel(r"$y/w$", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        cbar = plt.colorbar(heatmap, pad=0.01)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(r'$g_{2}^{(2)}$', fontsize=18, rotation=0, labelpad=15)
        plt.gca().set_aspect(1)
        plt.tight_layout()
        
        plt.savefig(f'confocal-scans/experiment_{light_structure_idx}_g1_g2.jpg', dpi=600, bbox_inches='tight')
        plt.close()
        
    
if __name__ == '__main__':
    
    main()