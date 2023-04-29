from scipy.special import hermite as physicistsHermite
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing as mp
from tem_utils import *
from parula import parula

matplotlib.rcParams['text.usetex'] = True

grid_x = 1750
grid_y = 1750

def main() -> None:
    
    # detector standard deviation
    
    detector_w = 1.
    
    # declare emitter positions
    
    # follows this pattern:
    # [[emitter_x, emitter_y], p_0_emitter]
    
    emitters = [
        [np.array([-0.6300,-0.1276]), 1.0000],
        [np.array([0.51460,-0.5573]), 0.3167]
    ]
    
    # extract just the positions
    
    emitter_positions = np.stack([emitter[0] for emitter in emitters])
    
    emitter_count = len(emitters)
    
    # obtain plot center
    
    plot_center = np.array([0., 0.])
    
    for emitter in emitters:
        
        plot_center += emitter[0]
        
    plot_center *= 1. / emitter_count
    
    # declare the light structures
    
    # follows this pattern:
    # [[m,n], w]
    # normalization coefficient is added automatically in next step
    
    light_structures = [
        [[1,1], 1.],
        [[3,2], 1.],
        [[0,1], 1.]
    ]
    
    # initialize their normalization factors
    
    for light_structure in light_structures:
        
        light_structure.append(normalizationCoefficient(light_structure[0][0], light_structure[0][1], light_structure[1]))
        
    # confocal raster of tem shape
    
    for light_structure_idx in range(len(light_structures)):
        
        light_structure = light_structures[light_structure_idx]
        
        x_range = (-1.5 * light_structure[1], 1.5 * light_structure[1])
        y_range = (-1.5 * light_structure[1], 1.5 * light_structure[1])
        
        y_linspace = np.linspace(*y_range, grid_y)
        x_linspace = np.linspace(*x_range, grid_x)
    
        arg_vec = [(*light_structure[0], y_val, grid_x, x_range) for y_val in y_linspace]
                
        multiprocessing_pool = mp.Pool(processes=mp.cpu_count())
        intensity_vals = np.stack(multiprocessing_pool.starmap(parallelTEM, arg_vec), axis=0)
        multiprocessing_pool.close()
        
        volume = np.trapz(
            y=np.asarray(
                [np.trapz(y=intensity_row, x=x_linspace) for intensity_row in intensity_vals[:]]
            ),
            x=y_linspace
        )
        
        plt.pcolormesh(x_linspace / light_structure[1], y_linspace / light_structure[1], intensity_vals / np.max(intensity_vals), cmap=parula)
        plt.xlabel(r"$x/w$", fontsize=18)
        plt.ylabel(r"$y/w$", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(r"$I\left(x,y\right)/I_{max}$", fontsize=18, rotation=-90, labelpad=20)
        plt.gca().set_aspect(1)
        plt.tight_layout()
        plt.savefig(f'tem-modes/experiment_{light_structure_idx}_tem_m_{light_structure[0][0]}_n_{light_structure[0][1]}.jpg', dpi=500, bbox_inches='tight')
        plt.close()
          
    # confocal raster of g_1 g_2 field
    
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
        heatmap = plt.pcolormesh(x_linspace, y_linspace, g_1_scan, cmap=parula)
        plt.scatter(emitter_positions[:,0], emitter_positions[:,1], c='k', marker='+', linewidths=1)
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
        plt.scatter(emitter_positions[:,0], emitter_positions[:,1], c='k', marker='+', linewidths=1)
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