import qclm
import numpy as np
import pathos.multiprocessing as pmp
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib
from parula import parula

matplotlib.rcParams['text.usetex'] = True

grid_x = 750
grid_y = 750

recordings = 200

detector = qclm.Detector

objective_position_noise = 2.
reading_noise_thresh = 0.01

def main() -> None:
    
    np.random.seed(1000)
    
    emitters = [
        qclm.Emitter(np.array([-0.15,0.2076]), 1.0),
        qclm.Emitter(np.array([0.18,-0.3573]), 0.3167)
    ]
    
    light_structures = [
        qclm.RectangularTEM(0, 0, 1., 0.),
        qclm.RectangularTEM(1, 0, 1., 0.),
        qclm.RectangularTEM(0, 1, 1., 0.),
    ]
    
    # use a uniform scan to get the objective location
    
    plot_center = np.array([0., 0.])
    
    x_range = (-0.5 * detector.w + plot_center[0],0.5 * detector.w + plot_center[0])
    y_range = (-0.5 * detector.w + plot_center[1],0.5 * detector.w + plot_center[1])

    y_linspace = np.linspace(*y_range, grid_y)
    x_linspace = np.linspace(*x_range, grid_x)
    
    uniform_illumination_scan = qclm.uniformIlluminationScan()
    
    arg_vec = [(emitters, uniform_illumination_scan, y_val, x_linspace, detector.w) for y_val in y_linspace]
    
    mp_pool = pmp.Pool(processes=pmp.cpu_count())
    g_1_g_2_concatenated = np.stack(mp_pool.starmap(qclm.parallelConfocalScan, arg_vec), axis=0)
    mp_pool.close()
    
    g_1_scan = g_1_g_2_concatenated[:,0:grid_x]
    g_2_scan = g_1_g_2_concatenated[:,grid_x:2*grid_x]
    
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
    plt.scatter(emitters[0].xy[0],emitters[0].xy[1], c='k', marker='x', s=40, linewidths=1.0)
    plt.scatter(emitters[1].xy[0],emitters[1].xy[1], c='k', marker='x', s=40, linewidths=1.0)
    plt.scatter(max_x,max_y, c='r', marker='x', s=40, linewidths=1.0, label=r'$\mathrm{Max\;Intensity}$')
    plt.scatter(xy_objective[0], xy_objective[1], c='r', marker='.', s=40, linewidths=1.0, label=r'$\mathrm{Objective\;Location}$')
    plt.legend(fontsize=18)
    plt.xlim(*x_range)
    plt.ylim(*y_range)
    plt.xlabel(r"$x/w$", fontsize=20)
    plt.ylabel(r"$y/w$", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    cbar = plt.colorbar(heatmap, pad=0.01)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(r'$g_{2}^{(1)}$', fontsize=20, rotation=0, labelpad=15)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig('xy_objective_g1.png', dpi=600, bbox_inches='tight')
    plt.close()
    
    heatmap = plt.pcolormesh(x_linspace, y_linspace, g_2_scan, cmap=parula, vmin=0., vmax=0.5)
    plt.scatter(emitters[0].xy[0],emitters[0].xy[1], c='k', marker='x', s=40, linewidths=1.0)
    plt.scatter(emitters[1].xy[0],emitters[1].xy[1], c='k', marker='x', s=40, linewidths=1.0)
    plt.scatter(max_x,max_y, c='r', marker='x', s=40, linewidths=1.0, label=r'$\mathrm{Max\;Intensity}$')
    plt.scatter(xy_objective[0], xy_objective[1], c='r', marker='.', s=40, linewidths=1.0, label=r'$\mathrm{Objective\;Location}$')
    plt.legend(fontsize=18)
    plt.xlim(*x_range)
    plt.ylim(*y_range)
    plt.xlabel(r"$x/w$", fontsize=20)
    plt.ylabel(r"$y/w$", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    cbar = plt.colorbar(heatmap, pad=0.01)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(r'$g_{2}^{(2)}$', fontsize=20, rotation=0, labelpad=15)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig('xy_objective_g2.png', dpi=600, bbox_inches='tight')
    plt.close()
    
    # get ground truth g_1 and g_2
    
    truth_g_1,truth_g_2 = qclm.groundTruthG1_G2(detector=detector, emitters=emitters, light_structures=light_structures, xy_objective=xy_objective)
    
    print(f'Truth G1:\n{truth_g_1}')
    print(f'Truth G2:\n{truth_g_2}')
    
    #============================================================================================================================================================    
    # optimization routine
    #============================================================================================================================================================

    def opt(recording_idx: int):
        
        np.random.seed(recording_idx)
        
        x0 = np.random.uniform(low=(-0.4),high=(0.4), size=(5,1)).flatten()
        x0[4] = 0.5
        
        # x0 = np.array([
        #     *emitters[0][0],*emitters[1][0],emitters[1][1]
        # ])
        
        # print(x0)
        
        print(f'Recording idx: {recording_idx}')
        
        noisy_g_1 = (truth_g_1 * np.random.uniform(low=(1-reading_noise_thresh),high=(1+reading_noise_thresh), size=(len(light_structures),1)))
        noisy_g_2 = (truth_g_2 * np.random.uniform(low=(1-reading_noise_thresh),high=(1+reading_noise_thresh), size=(len(light_structures),1)))
            
        optimization_lambda = lambda ps_vec: qclm.optimizeMe(ps_vec, detector, emitters, light_structures, xy_objective, noisy_g_1, noisy_g_2)
        
        xopt = np.zeros(shape=(1,5))
        
        # if method == 'fminsearch':
            
        xopt, _, iter, _, _ = scipy.optimize.fmin(func=optimization_lambda, x0=x0, disp=False, full_output=False, maxiter=200)
            
        # else:
            
        #     result = scipy.optimize.minimize(fun=optimization_lambda, x0=x0, method=method)
            
        #     xopt = result.x
            
        return xopt
    
    mp_pool = pmp.Pool(processes=pmp.cpu_count())
    mp_pool.map(opt, )

if __name__ == '__main__':
    
    main()