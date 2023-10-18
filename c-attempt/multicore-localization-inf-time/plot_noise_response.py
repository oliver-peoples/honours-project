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

cores = np.genfromtxt(os.path.join(path,'core_locations.csv'), delimiter=',', skip_header=1)

# pm_x = np.max(np.abs(cores[:,1]))
# pm_y = np.max(np.abs(cores[:,2]))

# pm = np.max([pm_x, pm_y]) * 1.1

pm = 2.2

g2_capable = np.genfromtxt(os.path.join(path,'g2_capable_indexes.csv'), delimiter=',', dtype=np.int8, skip_header=1)

if np.shape(g2_capable)[0] != np.shape(cores)[0]:
    g1_only = [ idx for idx in range(np.shape(cores)[0]) if idx not in g2_capable[:,1]]
    plt.scatter(cores[g1_only,1], cores[g1_only,2], c='black', marker='.', s=10)

# emitter_xys = np.genfromtxt(os.path.join(path,'emitter_parameter_log.csv'), delimiter=',', skip_header=1)

# for row in emitter_xys:
    
#     plt.plot([row[0],row[2]],[row[1],row[3]], c='magenta', linewidth=0.75, alpha=0.35)
#     plt.scatter([row[0],row[2]],[row[1],row[3]], c='magenta', marker='.', s=20, alpha=0.35)


plt.scatter(cores[g2_capable[:,1],1], cores[g2_capable[:,1],2], c='black', marker='+', linewidths=0.5, s=10)
    
plt.xlabel(r'$x$', fontsize=10)
plt.xlim(-pm,pm)
plt.xticks(fontsize=10)
plt.ylabel(r'$y$', fontsize=10)
plt.ylim(-pm,pm)
plt.xticks(fontsize=10)
plt.gca().set_aspect(1)
plt.tight_layout()
plt.gcf().set_figwidth(val=0.32 * 15.3978 * cm)
plt.savefig(os.path.join(path,f'core_locations.png'), dpi=600, bbox_inches='tight')
plt.close()

w_eff_data = np.genfromtxt(os.path.join(path,'w_eff_bar.csv'), delimiter=',', skip_header=1)

configs_per = np.shape(w_eff_data)[0]
noise_samples = np.shape(w_eff_data)[1]

bins = 70

noise_linspace = np.linspace(0, 20, noise_samples)
w_eff_linspace = np.linspace(0,1, bins)

noise_meshgrid, w_eff_meshgrid = np.meshgrid(noise_linspace, w_eff_linspace)

prevalence_data = np.ndarray((bins,noise_samples), dtype=np.float64)

for col_idx in range(noise_samples):
    
    counts, bin_ranges = np.histogram(w_eff_data[:,col_idx], bins, range=(0,1.))
    
    prevalence_data[:,col_idx] = 100. * counts / float(bins)
    
plt.pcolormesh(noise_meshgrid, w_eff_meshgrid, prevalence_data)
plt.show()