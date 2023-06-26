import qclm
import numpy as np
import pathos.multiprocessing as pmp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['text.usetex'] = True

grid_x = 750
grid_y = 750

recordings = 200

detector = qclm.Detector

objective_position_noise = 0.2

def main() -> None:
    
    emitters = [
        qclm.Emitter(np.array([-0.18,-0.1076]), 1.0),
        qclm.Emitter(np.array([-0.05,-0.5573]), 0.3167)
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
    plt.scatter(emitters[0][0][0],emitters[0][0][1], c='k', marker='x', s=20, linewidths=1.0)
    plt.scatter(emitters[1][0][0],emitters[1][0][1], c='k', marker='x', s=20, linewidths=1.0)
    plt.scatter(max_x,max_y, c='r', marker='x', s=20, linewidths=1.0, label=r'$\mathrm{Max\;Intensity}$')
    plt.scatter(xy_objective[0], xy_objective[1], c='r', marker='.', s=20, linewidths=1.0, label=r'$\mathrm{Objective\;Location}$')
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
    plt.savefig('xy_objective.png', dpi=600, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    
    main()