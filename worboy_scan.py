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
    
    g_1_scan = np.genfromtxt('g_1_confocal_scan_intensities.csv', delimiter=',')

    heatmap = plt.pcolormesh(x_linspace, y_linspace, g_1_scan, cmap=parula)
    plt.scatter(emitter_positions[:,0], emitter_positions[:,1], c='k', marker='+', linewidths=1)
    plt.xlabel(r"$x/w$", fontsize=18)
    plt.ylabel(r"$y/w$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cbar = plt.colorbar(heatmap, pad=0.01)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(r'$g_{2}^{(1)}$', fontsize=18, rotation=0, labelpad=15)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(f'worboy_g_1_scan.jpg', dpi=600, bbox_inches='tight')
    plt.close()
    
    # plt g_2
    
    g_2_scan = np.genfromtxt('g_2_confocal_scan_intensities.csv', delimiter=',')
              
    heatmap = plt.pcolormesh(x_linspace, y_linspace, g_2_scan, cmap=parula)
    plt.scatter(emitter_positions[:,0], emitter_positions[:,1], c='k', marker='+', linewidths=1)
    # manual_locations = []
    contour_lines = plt.contour(x_linspace, y_linspace, g_2_scan, colors='k', levels=np.linspace(0.2,0.45,6))
    plt.gca().clabel(contour_lines, inline=True, fontsize=10)
    plt.xlabel(r"$x/w$", fontsize=18)
    plt.ylabel(r"$y/w$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cbar = plt.colorbar(heatmap, pad=0.01)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(r'$g_{2}^{(2)}$', fontsize=18, rotation=0, labelpad=15)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(f'worboy_g_2_scan.jpg', dpi=600, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    
    main()