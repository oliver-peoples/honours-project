# from scipy.special import hermite as physicistsHermite
import numpy as np
# from parula import parula
import matplotlib.pyplot as plt
import matplotlib
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
# from scipy.spatial import ConvexHull, convex_hull_plot_2d

cm = 1/2.54

path = os.path.dirname(__file__)

matplotlib.rcParams['text.usetex'] = True

def main(data_path) -> None:

    cores = np.genfromtxt(os.path.join(path,data_path,'core_locations.csv'), delimiter=',', skip_header=1)

    pm = 2.2

    g2_capable = np.genfromtxt(os.path.join(path,data_path,'g2_capable_indexes.csv'), delimiter=',', dtype=np.int8, skip_header=1)

    g1_only = []
    
    if np.shape(g2_capable)[0] != np.shape(cores)[0]:
        g1_only = [ idx for idx in range(np.shape(cores)[0]) if idx not in g2_capable[:,1]]
        plt.scatter(cores[g1_only,1], cores[g1_only,2], c='black', marker='.', s=10)
        
    # for core in cores:
    #     # fix radius here
        
    #     shape = RegularPolygon((core[1], core[2]), numVertices=6, radius=0.5, alpha=0.2, edgecolor='k')
        
    #     shape = Circle((core[1],core[2]), radius=0.45, alpha=0.2, edgecolor='k')
        
    #     plt.gca().add_patch(shape)

    plt.scatter(cores[g2_capable[:,1],1], cores[g2_capable[:,1],2], c='black', marker='+', linewidths=0.5, s=10)
    # plt.scatter(emitter_xy[:,1], emitter_xy[:,2], c='blue', marker='x', linewidths=0.5, s=10)
    plt.xlabel(r'$x/\sigma$', fontsize=10)
    plt.xlim(-pm,pm)
    plt.xticks(fontsize=10)
    plt.ylabel(r'$y/\sigma$', fontsize=10)
    plt.ylim(-pm,pm)
    plt.xticks(fontsize=10)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.gcf().set_figwidth(val=0.32 * 15.3978 * cm)
    plt.savefig(os.path.join(path,data_path,f'core_locations.png'), dpi=600, bbox_inches='tight')
    plt.close()

    w_eff_data = np.genfromtxt(os.path.join(path,data_path,'w_eff_bar.csv'), delimiter=',', skip_header=1)

    configs_per = np.shape(w_eff_data)[0]
    noise_samples = np.shape(w_eff_data)[1]

    bins = 70

    noise_linspace = np.linspace(0, 20, noise_samples)
    w_eff_linspace = np.linspace(0,1, bins)

    noise_meshgrid, w_eff_meshgrid = np.meshgrid(noise_linspace, w_eff_linspace)

    prevalence_data = np.ndarray((bins,noise_samples), dtype=np.float64)

    for col_idx in range(noise_samples):
        
        counts, bin_ranges = np.histogram(w_eff_data[:,col_idx], bins, range=(0,1.0))
        
        prevalence_data[:,col_idx] = 100 * counts / np.sum(counts)
        
    # plt.show()
    
    plt.figure()
    plt.gcf().set_figwidth(val=0.49 * 15.3978 * cm)
    plt.gcf().set_figheight(val=0.4 * 15.3978 * cm)
    plt.pcolormesh(noise_meshgrid, w_eff_meshgrid, prevalence_data, vmax=15, vmin=0)
    cbar = plt.colorbar(pad=0.01)
    cbar.ax.tick_params(labelsize=8)
    ticks = [0,5,10,15]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([r'$' + str(tick) + r'$' for tick in ticks])
    cbar.set_label(r"$\mathrm{Relative\;Frequency\;[\%]}$", fontsize=8, rotation=90, labelpad=2)
    plt.xlabel(r'$\eta\mathrm{\;[\%]}$', fontsize=8)
    plt.xticks(fontsize=8)
    plt.ylabel(r'$\bar{w}_{eff} \mathrm{\;[\sigma]}$', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(path,data_path,'relative_frequency.png'), dpi=500, bbox_inches='tight')
    plt.close()
    # plt.close()
    # prevalence_data = np.flip(prevalence_data, axis=0)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # im = ax.imshow(prevalence_data)
    # ax.set_xlabel('xlabel')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=-2)
    # cbar = plt.colorbar(im, cax=cax)
    # cbar.ax.tick_params(labelsize=8)
    # cbar.set_label(r"$Relative Prevalence$", fontsize=8, rotation=-90, labelpad=15)
    # forceAspect(ax,aspect=1)
    # plt.gcf().set_figwidth(val=0.49 * 15.3978 * cm)
    # plt.gcf().savefig(os.path.join(path,'relative_prevalances.png'), dpi=500, bbox_inches='tight')


    # plt.figure()
    # plt.pcolormesh(noise_meshgrid, w_eff_meshgrid, prevalence_data)
    # # plt.title(r'$r=' + f'{r:.3f}' + r'\sigma$', fontsize=8)
    # ax = plt.gca()
    # # im = ax.imshow(prevalence_data, interpolation='none', extent=[0,np.max(noise_linspace),0.0,0.5])
    # ax.set_aspect(np.max(noise_linspace) / (1 * 0.5))
    
    # # plt.scatter(e_1_xy[0],e_1_xy[1], c='blue', marker='x', linewidths=0.5, s=10)
    # # plt.scatter(e_2_xy[0],e_2_xy[1], c='blue', marker='x', linewidths=0.5, s=10)
    # # 
    # plt.xlabel(r"$\eta$", fontsize=8, labelpad=1)
    # plt.ylabel(r"$\bar{w}_{eff}$", fontsize=8, labelpad=3)
    # plt.gca().tick_params(labelsize=8)
    # plt.tight_layout()
    # plt.savefig(os.path.join(path,'relative_prevalances.png'), dpi=500, bbox_inches='tight')
    # plt.close()
    
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
    
if __name__ == '__main__':
    
    data_path = sys.argv[1]
    
    main(data_path=data_path)