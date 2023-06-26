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
from trilateration import *

matplotlib.rcParams['text.usetex'] = True

# ['fminsearch','nelder-meade']

# method = 'fminsearch'
method = 'Nelder-Mead'

#============================================================================================================================================================
# initialization data
#============================================================================================================================================================

grid_x = 750
grid_y = 750

recordings = 200
# pm_thresh = 0.1 # emitter start location noise
noise_thresh = 0.01
range_modifier = 0.5
objective_position_noise = 0.2

# detector standard deviation

detector_w = 1.    # declare the light structures

emitters = [
    [np.array([-0.18,-0.1076]) * range_modifier, 1.0000],
    [np.array([-0.05,-0.5573]) * range_modifier, 0.3167]
]

emitter_separation = np.linalg.norm(emitters[0][0]-emitters[1][0])
print(f'Emitter separation: {emitter_separation}')

# obtain plot center

plot_center = np.array([0., 0.])

light_structures = [
    [[0,0], 1., 0.],
    [[1,0], 0.5, 0.],
    [[0,1], 0.5, 0.],
    # [[1,1], 1., 0.]
]

# initialize their normalization factors

for light_structure in light_structures:
    
    light_structure.append(normalizationCoefficient(light_structure[0][0], light_structure[0][1], light_structure[1]))
    
#============================================================================================================================================================
# get objective position
#============================================================================================================================================================

x_range = (-0.5 * detector_w + plot_center[0],0.5 * detector_w + plot_center[0])
y_range = (-0.5 * detector_w + plot_center[1],0.5 * detector_w + plot_center[1])

y_linspace = np.linspace(*y_range, grid_y)
x_linspace = np.linspace(*x_range, grid_x)

uniform_illumination_scan = [[0,0], 1000000, 0]
uniform_illumination_scan.append(normalizationCoefficient(uniform_illumination_scan[0][0], uniform_illumination_scan[0][1], uniform_illumination_scan[1]))

arg_vec = [(emitters, uniform_illumination_scan, y_val, x_linspace, detector_w) for y_val in y_linspace]
    
multiprocessing_pool = mp.Pool(processes=mp.cpu_count())
g_1_g_2_concatenated = np.stack(multiprocessing_pool.starmap(parallelConfocalScan, arg_vec), axis=0)
multiprocessing_pool.close()

g_1_scan = g_1_g_2_concatenated[:,0:grid_x]
g_2_scan = g_1_g_2_concatenated[:,grid_x:2*grid_x]

# fig, ax = plt.subplots(2, 1, figsize=(8, 8))

# plt.subplot(2,1,1)

max_indices = np.unravel_index(np.argmax(g_1_scan, axis=None), g_1_scan.shape)

max_y = y_linspace[max_indices[0]]
max_x = x_linspace[max_indices[1]]

g_1_scan = g_1_g_2_concatenated[:,0:grid_x]
g_2_scan = g_1_g_2_concatenated[:,grid_x:2*grid_x]

max_indices = np.unravel_index(np.argmax(g_1_scan, axis=None), g_1_scan.shape)

max_y = y_linspace[max_indices[0]]
max_x = x_linspace[max_indices[1]]

xy_objective = np.array([max_x,max_y], dtype=np.float64)

xy_objective *= np.random.uniform(low=(1-objective_position_noise),high=(1+objective_position_noise), size=(2,))

heatmap = plt.pcolormesh(x_linspace, y_linspace, g_1_scan / np.max(g_1_scan), cmap=parula)
plt.scatter(emitters[0][0][0],emitters[0][0][1], c='k', marker='x', s=20, linewidths=1.0)
plt.scatter(emitters[1][0][0],emitters[1][0][1], c='k', marker='x', s=20, linewidths=1.0)
plt.scatter(max_x,max_y, c='r', marker='x', s=20, linewidths=1.0, label=r'$\mathrm{Max\;Intensity}$')
plt.scatter(xy_objective[0], xy_objective[1], c='r', marker='.', s=20, linewidths=1.0, label=r'$\mathrm{Objective}$')
plt.legend(fontsize=16)
plt.xlim(*x_range)
plt.ylim(*y_range)
plt.xlabel(r"$x/w$", fontsize=18)
plt.ylabel(r"$y/w$", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
cbar = plt.colorbar(heatmap, pad=0.01)
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$g_{2}^{(1)}$', fontsize=18, rotation=0, labelpad=15)
plt.gca().set_aspect(1)
plt.tight_layout()
plt.savefig('trilateration-out/xy_objective.png', dpi=600, bbox_inches='tight')
plt.close()

#============================================================================================================================================================    
# get truth g_1 and g_2 values
#============================================================================================================================================================

truth_g_1,truth_g_2 = groundTruthG1_G2(detector_w=detector_w, emitters=emitters, light_structures=light_structures, xy_objective=xy_objective)

#============================================================================================================================================================    
# optimization routine
#============================================================================================================================================================

def opt(recording_idx: int):
    
    np.random.seed(recording_idx)
    
    x0 = np.random.uniform(low=(-emitter_separation),high=(emitter_separation), size=(5,1)).flatten()
    x0[4] = 0.5
    
    # x0 = np.array([
    #     *emitters[0][0],*emitters[1][0],emitters[1][1]
    # ])
    
    # print(x0)
    
    print(f'Recording idx: {recording_idx}')
    
    noisy_g_1 = (truth_g_1 * np.random.uniform(low=(1-noise_thresh),high=(1+noise_thresh), size=(len(light_structures),1)))
    noisy_g_2 = (truth_g_2 * np.random.uniform(low=(1-noise_thresh),high=(1+noise_thresh), size=(len(light_structures),1)))
        
    optimization_lambda = lambda ps_vec: sumSquareDifferences(ps_vec, noisy_g_1, noisy_g_2, detector_w, emitters, light_structures, xy_objective)
    
    xopt = np.zeros(shape=(1,5))
    
    if method == 'fminsearch':
        
        xopt, _, iter, _, _ = scipy.optimize.fmin(func=optimization_lambda, x0=x0, disp=False, full_output=False, maxiter=200)
        
    else:
        
        result = scipy.optimize.minimize(fun=optimization_lambda, x0=x0, method=method)
        
        xopt = result.x
        
    return xopt

def main() -> None:
    
    truth = np.array([*emitters[0][0],*emitters[1][0],emitters[1][1]/emitters[0][1]], dtype=np.float64)
    
    xopt = np.zeros(shape=(recordings + 2,5))
    
    xopt[0,:] = truth
    
    mp_pool = mp.Pool(processes=mp.cpu_count())
    
    xopt_list = mp_pool.map(opt, range(recordings))
    mp_pool.close()
    
    for recording_idx in range(recordings):
        
        xopt[recording_idx + 2] = xopt_list[recording_idx]
    
    # [xopt[recording_idx + 2,:] := xopt_list[recording_idx,:] for recording_idx in range(recordings)]
    
    # for recording_idx in range(recordings):
        
        
        
    #     # if xopt[recording_idx + 2,4] > 1:
            
    #     #     xopt[recording_idx + 2,4] = 1/xopt[recording_idx + 2,4]
            
    #     #     old_e1 = xopt[recording_idx + 2,0:2]
    #     #     xopt[recording_idx + 2,0:2] = xopt[recording_idx + 2,2:4]
    #     #     xopt[recording_idx + 2,2:4] = old_e1
        
    xopt[1,:] = np.mean(xopt[2:,:], 0)

    plt.scatter(xopt[:,0],xopt[:,1], c='b', s=2., marker='.')
    plt.scatter(xopt[:,2],xopt[:,3], c='r', s=2., marker='.')
    plt.scatter(xy_objective[0], xy_objective[1], c='k', marker='+', s=20, linewidths=1.0, label=r'$\mathrm{Objective}$')
    plt.scatter(emitters[0][0][0],emitters[0][0][1], c='k', marker='x', s=20, linewidths=0.5)
    plt.scatter(emitters[1][0][0],emitters[1][0][1], c='k', marker='x', s=20, linewidths=0.5)
    plt.scatter(xopt[0,0],xopt[0,1], c='k', s=20, marker='x', linewidths=0.5)
    plt.scatter(xopt[0,2],xopt[0,3], c='k', s=20, marker='x', linewidths=0.5)    
    plt.xlim(*x_range)
    plt.ylim(*y_range)
    plt.xlabel(r"$x/w$", fontsize=18)
    plt.ylabel(r"$y/w$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig('trilateration-out/xopt.png', dpi=600, bbox_inches='tight')
    
    np.savetxt('trilateration-out/xopt.csv', xopt, delimiter=',')
    
    for light_structure_idx in range(len(light_structures)):
        
        light_structure = light_structures[light_structure_idx]
        
        # x_range = (-3 * detector_w + plot_center[0],3 * detector_w + plot_center[0])
        # y_range = (-3 * detector_w + plot_center[1],3 * detector_w + plot_center[1])
        
        # y_linspace = np.linspace(*y_range, grid_y)
        # x_linspace = np.linspace(*x_range, grid_x)
        
        arg_vec = [(emitters, light_structure, y_val, x_linspace, detector_w) for y_val in y_linspace]
        
        multiprocessing_pool = mp.Pool(processes=mp.cpu_count())
        g_1_g_2_concatenated = np.stack(multiprocessing_pool.starmap(parallelConfocalScan, arg_vec), axis=0)
        multiprocessing_pool.close()
        
        g_1_scan = g_1_g_2_concatenated[:,0:grid_x]
        g_2_scan = g_1_g_2_concatenated[:,grid_x:2*grid_x]
        
        fig, ax = plt.subplots(2, 1, figsize=(8, 8))
        
        max_indices = np.unravel_index(np.argmax(g_1_scan, axis=None), g_1_scan.shape)
        
        max_y = y_linspace[max_indices[0]]
        max_x = x_linspace[max_indices[1]]
        
        plt.subplot(2,1,1)
        
        exec('plt.title(r"$\mathbf{TEM}_{' + str(light_structure[0][0]) + ',' + str(light_structure[0][1]) + '}$", fontsize=22)')
        heatmap = plt.pcolormesh(x_linspace, y_linspace, g_1_scan, cmap=parula)
        plt.scatter(xy_objective[0], xy_objective[1], c='r', marker='.', s=20, linewidths=1.0, label=r'$\mathrm{Objective}$')
        # plt.scatter(xopt[:,0],xopt[:,1], c='b', s=1., marker='.')
        # plt.scatter(xopt[:,2],xopt[:,3], c='r', s=1., marker='.')
        plt.scatter(emitters[0][0][0],emitters[0][0][1], c='k', marker='x', s=20, linewidths=1.0)
        plt.scatter(emitters[1][0][0],emitters[1][0][1], c='k', marker='x', s=20, linewidths=1.0)
        # plt.scatter(max_x,max_y, c='r', marker='.', s=20, linewidths=1.0)
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
        cbar.set_label(r'$g_{2}^{(1)}$', fontsize=18, rotation=0, labelpad=15)
        plt.gca().set_aspect(1)
        plt.tight_layout()
        
        # plt g_2
        
        plt.subplot(2,1,2)            
        heatmap = plt.pcolormesh(x_linspace, y_linspace, g_2_scan, cmap=parula)
        plt.scatter(xy_objective[0], xy_objective[1], c='r', marker='.', s=20, linewidths=1.0, label=r'$\mathrm{Objective}$')
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
        
        plt.savefig(f'trilateration-out/trilateration_out_g1_g2_experiment_{light_structure_idx}.jpg', dpi=600, bbox_inches='tight')
        plt.close()
        
    
if __name__ == '__main__':
    
    main()