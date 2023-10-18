# from scipy.special import hermite as physicistsHermite
import numpy as np
# from parula import parula
import matplotlib.pyplot as plt
import matplotlib
import os
# from scipy.spatial import ConvexHull, convex_hull_plot_2d

cm = 1/2.54

path = os.path.dirname(__file__)

matplotlib.rcParams['text.usetex'] = True

def main() -> None:

    conf_frac = 1 - 1/np.sqrt(np.exp(1))
    conf_frac *= 1

    cores = np.genfromtxt(os.path.join(path,'core_locations.csv'), delimiter=',', skip_header=1)

    pm_x = np.max(np.abs(cores[:,1]))
    pm_y = np.max(np.abs(cores[:,2]))

    pm = np.max([pm_x, pm_y]) * 1.1

    g2_capable = np.genfromtxt(os.path.join(path,'g2_capable_indexes.csv'), delimiter=',', dtype=np.int8, skip_header=1)
    g1_only = [ idx for idx in range(np.shape(cores)[0]) if idx not in g2_capable[:,1]]

    plt.scatter(cores[g1_only,1], cores[g1_only,2], c='black', marker='.', s=10)
    plt.scatter(cores[g2_capable[:,1],1], cores[g2_capable[:,1],2], c='black', marker='+', linewidths=0.5, s=10)
    plt.xlabel(r'$x$', fontsize=12)
    plt.xlim(-pm,pm)
    plt.xticks(fontsize=12)
    plt.ylabel(r'$y$', fontsize=12)
    plt.ylim(-pm,pm)
    plt.xticks(fontsize=12)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(os.path.join(path,'core_locations.png'), dpi=500, bbox_inches='tight')
    plt.close()

    x1s = np.genfromtxt(os.path.join(path,'x1s.csv'), delimiter=',', skip_header=1)
    x2s = np.genfromtxt(os.path.join(path,'x2s.csv'), delimiter=',', skip_header=1)

    emitter_xy = np.genfromtxt(os.path.join(path,'emitter_xy.csv'), delimiter=',', skip_header=1)

    plt.scatter(x2s[:,1],x2s[:,2], c='magenta', marker='.', s=1)
    plt.scatter(x1s[:,1],x1s[:,2], c='cyan', marker='.', s=1)
    
    x1s_convex_hull = np.genfromtxt(os.path.join(path, 'x1s_convex_hull.csv'), delimiter=',', skip_header=1)
    
    plt.plot(x1s_convex_hull[:,1],x1s_convex_hull[:,2], c='black', linewidth=0.5)
    
    x2s_convex_hull = np.genfromtxt(os.path.join(path, 'x2s_convex_hull.csv'), delimiter=',', skip_header=1)
    
    plt.plot(x2s_convex_hull[:,1],x2s_convex_hull[:,2], c='black', linewidth=0.5)


    # rr_1 = np.sqrt((x1s[:,1] - np.mean(x1s[:,1]))**2 + (x1s[:,2] - np.mean(x1s[:,2]))**2);
    # r_tab_1 = np.zeros((np.shape(rr_1)[0], 3))
    # r_tab_1[:,0:2] = x1s[:,1:]
    # r_tab_1[:,2] = rr_1
    # r_tab_idx = np.argsort(r_tab_1[:,2])
    # r_tab_1 = r_tab_1[r_tab_idx,:]
    # # SortTab1 = (RTab1,3);
    # frac_boundary = int(np.ceil(conf_frac*np.shape(rr_1)[0]))
    # r_tab_1 = r_tab_1[0:frac_boundary,:]
    
    # plotConvexHull(r_tab_1[:,0:2])
    
    # rr_2 = np.sqrt((x2s[:,1] - np.mean(x2s[:,1]))**2 + (x2s[:,2] - np.mean(x2s[:,2]))**2);
    # r_tab_2 = np.zeros((np.shape(rr_2)[0], 3))
    # r_tab_2[:,0:2] = x2s[:,1:]
    # r_tab_2[:,2] = rr_2
    # r_tab_idx = np.argsort(r_tab_2[:,2])
    # r_tab_2 = r_tab_2[r_tab_idx,:]
    # # SortTab1 = (RTab1,3);
    # frac_boundary = int(np.ceil(conf_frac*np.shape(rr_2)[0]))
    # r_tab_2 = r_tab_2[0:frac_boundary,:]
    
    # plotConvexHull(r_tab_2[:,0:2])

    plt.scatter(cores[g1_only,1], cores[g1_only,2], c='black', marker='.', s=10)
    plt.scatter(cores[g2_capable[:,1],1], cores[g2_capable[:,1],2], c='black', marker='+', linewidths=0.5, s=10)
    plt.scatter(emitter_xy[:,1], emitter_xy[:,2], c='blue', marker='x', linewidths=0.5, s=10)
    plt.xlabel(r'$x$', fontsize=12)
    plt.xlim(-2.1,2.1)
    plt.xticks(fontsize=12)
    plt.ylabel(r'$y$', fontsize=12)
    plt.ylim(-2.1,2.1)
    plt.xticks(fontsize=12)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.gcf().set_figwidth(val=0.49 * 15.3978 * cm)
    plt.savefig(os.path.join(path,f'emitter_localizations_{np.shape(x1s)[0]}.png'), dpi=600, bbox_inches='tight')
    plt.close()
    
# def plotConvexHull(ordered_points):
    
#     hull = ConvexHull(ordered_points)
    
#     for simplex in hull.simplices:

#         plt.plot(ordered_points[simplex, 0], ordered_points[simplex, 1], color='black', linewidth=0.5)
    
if __name__ == '__main__':
    
    main()