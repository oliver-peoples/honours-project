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

recordings = 1000

detector = qclm.Detector

objective_position_noise = 0.1
reading_noise_thresh = 0.05

def main() -> None:
    
    np.random.seed(1000)
    
    emitters = [
        qclm.Emitter(0.15 * np.array([-0.6300,-0.1276]), 1.0),
        qclm.Emitter(0.15 * np.array([0.51460,-0.5573]), 0.71367)
    ]
    
    print(f'Emitter separation: {np.linalg.norm(emitters[0].xy - emitters[1].xy)}')
    
    light_structures = [
        qclm.RectangularTEM(0, 0, 1., 0.),
        qclm.RectangularTEM(0, 1, 1., 0.),
        qclm.RectangularTEM(1, 0, 1., 0.),
        qclm.RectangularTEM(1, 1, 1., 0.),
        qclm.RectangularTEM(2, 1, 1., 0.),
        qclm.RectangularTEM(1, 2, 1., 0.),
        # qclm.RectangularTEM(1, 2, 0.5, 0.),
    ]
    
    # use a uniform scan to get the objective location
    
    plot_center = np.array([0.5 * (emitters[0].xy[0] + emitters[1].xy[0]), 0.5 * (emitters[0].xy[1] + emitters[1].xy[1])])
    
    print(f'Plot center: {plot_center}')
    
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
    # alpha optimization routine
    #============================================================================================================================================================

    def optAlpha(recording_idx: int):
        
        # np.random.seed(recording_idx)
        
        x_0 = np.zeros(shape=(5,))
        
        x_0[:2] = np.random.uniform(low=(-0.2),high=(0.2), size=(1,2)).flatten()
        x_0[2:4] = xy_objective - (x_0[:2] - xy_objective)
        x_0[4] = 0.5
        
        # x_0 = np.array([
        #     *emitters[0].xy,*emitters[1].xy,emitters[1].relative_brightness
        # ])
    
        
        # print(x_0)
        
        print(f'Recording idx: {recording_idx}')
        
        noisy_g_1 = (truth_g_1 * np.random.uniform(low=(1-reading_noise_thresh),high=(1+reading_noise_thresh), size=(len(light_structures),1)))
        noisy_g_2 = (truth_g_2 * np.random.uniform(low=(1-reading_noise_thresh),high=(1+reading_noise_thresh), size=(len(light_structures),1)))
            
        optimization_lambda = lambda ps_vec: qclm.optimizeMe(ps_vec, detector, emitters, light_structures, xy_objective, noisy_g_1, noisy_g_2)
        
        x_opt = np.zeros(shape=(1,5))
        
        # if method == 'fminsearch':
            
        x_opt, _, iter, _, _ = scipy.optimize.fmin(func=optimization_lambda, x0=x_0, disp=False, full_output=True, maxiter=400)
            
        # else:
            
        #     result = scipy.optimize.minimize(fun=optimization_lambda, x_0=x_0, method=method)
            
        #     x_opt = result.x
            
        return x_opt
    
    # mp_pool = pmp.Pool(processes=1)
    mp_pool = pmp.Pool(processes=int(pmp.cpu_count() / 2))
    # mp_pool = pmp.Pool(processes=pmp.cpu_count())
    x_opt_list = mp_pool.map(optAlpha, list(range(recordings)))
    mp_pool.close()
    
    x_opt = np.zeros(shape=(recordings + 2,5))
    
    x_opt[0] = np.array([
        *emitters[0].xy,*emitters[1].xy,emitters[1].relative_brightness
    ])
    
    for recording_idx in range(recordings):
        
        x_opt[recording_idx + 2] = x_opt_list[recording_idx]
        
    x_opt[1,:] = np.mean(x_opt[2:,:], 0)

    plt.scatter(x_opt[2:,0],x_opt[2:,1], c='b', s=2., marker='.')
    plt.scatter(x_opt[2:,2],x_opt[2:,3], c='r', s=2., marker='.')
    plt.scatter(xy_objective[0], xy_objective[1], c='r', marker='.', s=40, linewidths=1.0, label=r'$\mathrm{Objective\;Location}$')
    plt.scatter(max_x,max_y, c='r', marker='x', s=40, linewidths=1.0, label=r'$\mathrm{Max\;Intensity}$')
    plt.scatter(emitters[0].xy[0],emitters[0].xy[1], c='k', marker='x', s=40, linewidths=1.0)
    plt.scatter(emitters[1].xy[0],emitters[1].xy[1], c='k', marker='x', s=40, linewidths=1.0)
    plt.scatter(x_opt[1,0],x_opt[1,1], c='k', s=40, marker='+', linewidths=0.5)
    plt.scatter(x_opt[1,2],x_opt[1,3], c='k', s=40, marker='+', linewidths=0.5)    
    plt.xlim(*x_range)
    plt.ylim(*y_range)
    plt.xlabel(r"$x/w$", fontsize=18)
    plt.ylabel(r"$y/w$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig('x_opt_prelim.png', dpi=600, bbox_inches='tight')
    
    np.savetxt('x_opt_prelim.csv', x_opt, delimiter=',')
    
    opt_alpha = x_opt[1,-1]
    
    print(f'Estimated alpha: {opt_alpha}, actual alpha: {emitters[1].relative_brightness}')
    
    #============================================================================================================================================================    
    # position optimization routine
    #============================================================================================================================================================

    def optPosition(recording_idx: int):
        
        # np.random.seed(recording_idx)
        
        x_0 = np.zeros(shape=(4,))
        
        x_0[:2] = np.random.uniform(low=(-0.2),high=(0.2), size=(1,2)).flatten()
        x_0[2:4] = xy_objective - (x_0[:2] - xy_objective)
        
        # x_0 = np.array([
        #     *emitters[0].xy,*emitters[1].xy,emitters[1].relative_brightness
        # ])
    
        
        # print(x_0)
        
        print(f'Recording idx: {recording_idx}')
        
        noisy_g_1 = (truth_g_1 * np.random.uniform(low=(1-reading_noise_thresh),high=(1+reading_noise_thresh), size=(len(light_structures),1)))
        noisy_g_2 = (truth_g_2 * np.random.uniform(low=(1-reading_noise_thresh),high=(1+reading_noise_thresh), size=(len(light_structures),1)))
            
        optimization_lambda = lambda ps_vec: qclm.optimizeMe(np.array([*ps_vec,opt_alpha]), detector, emitters, light_structures, xy_objective, noisy_g_1, noisy_g_2)
        
        x_opt = np.zeros(shape=(1,4))
        
        # if method == 'fminsearch':
            
        x_opt, _, iter, _, _ = scipy.optimize.fmin(func=optimization_lambda, x0=x_0, disp=False, full_output=True, maxiter=400)
            
        # else:
            
        #     result = scipy.optimize.minimize(fun=optimization_lambda, x_0=x_0, method=method)
            
        #     x_opt = result.x
            
        return x_opt
    
    # mp_pool = pmp.Pool(processes=1)
    mp_pool = pmp.Pool(processes=int(pmp.cpu_count() / 2))
    # mp_pool = pmp.Pool(processes=pmp.cpu_count())
    x_opt_list = mp_pool.map(optPosition, list(range(recordings)))
    mp_pool.close()
    
    x_opt = np.zeros(shape=(recordings + 2,4))
    
    x_opt[0] = np.array([
        *emitters[0].xy,*emitters[1].xy
    ])
    
    for recording_idx in range(recordings):
        
        x_opt[recording_idx + 2] = x_opt_list[recording_idx]
        
    x_opt[1,:] = np.mean(x_opt[2:,:], 0)

    plt.scatter(x_opt[2:,0],x_opt[2:,1], c='b', s=2., marker='.')
    plt.scatter(x_opt[2:,2],x_opt[2:,3], c='r', s=2., marker='.')
    plt.scatter(xy_objective[0], xy_objective[1], c='r', marker='.', s=40, linewidths=1.0, label=r'$\mathrm{Objective\;Location}$')
    plt.scatter(max_x,max_y, c='r', marker='x', s=40, linewidths=1.0, label=r'$\mathrm{Max\;Intensity}$')
    plt.scatter(emitters[0].xy[0],emitters[0].xy[1], c='k', marker='x', s=40, linewidths=1.0)
    plt.scatter(emitters[1].xy[0],emitters[1].xy[1], c='k', marker='x', s=40, linewidths=1.0)
    plt.scatter(x_opt[1,0],x_opt[1,1], c='k', s=40, marker='+', linewidths=0.5)
    plt.scatter(x_opt[1,2],x_opt[1,3], c='k', s=40, marker='+', linewidths=0.5)    
    plt.xlim(*x_range)
    plt.ylim(*y_range)
    plt.xlabel(r"$x/w$", fontsize=18)
    plt.ylabel(r"$y/w$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig('x_opt_position.png', dpi=600, bbox_inches='tight')
    
    np.savetxt('x_opt_position.csv', x_opt, delimiter=',')

if __name__ == '__main__':
    
    main()