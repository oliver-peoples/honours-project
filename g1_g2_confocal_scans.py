import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from parula import parula

def main() -> None:
    
    # basic requirements for plotting
    
    emitter_positions = np.genfromtxt('emitter_positions.csv', delimiter=',')
    
    x_linspace = np.genfromtxt('x_linspace.csv', delimiter=',')
    y_linspace = np.genfromtxt('y_linspace.csv', delimiter=',')
    
    # plot g_1
    
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    
    g_1_scan = np.genfromtxt('g_1_confocal_scan_intensities.csv', delimiter=',')
    
    plt.subplot(2,1,1)
    heatmap = plt.pcolormesh(x_linspace, y_linspace, g_1_scan, cmap=parula)
    plt.scatter(emitter_positions[:,0], emitter_positions[:,1], c='k', marker='+', linewidths=1)
    plt.xlabel(r"$x/w$", fontsize=16)
    plt.ylabel(r"$y/w$", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    cbar = plt.colorbar(heatmap)
    cbar.set_label(r'$g_{2}^{(1)}$', fontsize=16, rotation=0, labelpad=15)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    # plt.savefig(f'g_1_scan.png', dpi=1200, bbox_inches='tight')
    # plt.close()
    
    # plt g_2
    
    g_2_scan = np.genfromtxt('g_2_confocal_scan_intensities.csv', delimiter=',')
    
    plt.subplot(2,1,2)            
    heatmap = plt.pcolormesh(x_linspace, y_linspace, g_2_scan, cmap=parula)
    plt.scatter(emitter_positions[:,0], emitter_positions[:,1], c='k', marker='+', linewidths=1)
    plt.xlabel(r"$x/w$", fontsize=16)
    plt.ylabel(r"$y/w$", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    cbar = plt.colorbar(heatmap)
    cbar.set_label(r'$g_{2}^{(2)}$', fontsize=16, rotation=0, labelpad=15)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    
    plt.savefig(f'confocal_scan.png', dpi=1200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    
    main()