# from scipy.special import hermite as physicistsHermite
import numpy as np
# from parula import parula
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import RegularPolygon, Circle
import os

# from scipy.spatial import ConvexHull, convex_hull_plot_2d

cm = 1/2.54

path = os.path.dirname(__file__)

matplotlib.rcParams['text.usetex'] = True

radial_strat = np.genfromtxt(os.path.join(path,'data','w_eff_bar_2_19_1,3,5,7,11,15.csv'), delimiter=',', skip_header=1)
shift_strat = np.genfromtxt(os.path.join(path,'data','w_eff_bar_2_19_1,3,5,9,13,17.csv'), delimiter=',', skip_header=1)

# radial_counts

configs_per = np.shape(radial_strat)[0]
noise_samples = np.shape(radial_strat)[1]

bins = 70

noise_linspace = np.linspace(0, 20, noise_samples)
w_eff_linspace = np.linspace(0,1, bins)

noise_meshgrid, w_eff_meshgrid = np.meshgrid(noise_linspace, w_eff_linspace)

radial_frequencies = np.ndarray((bins,noise_samples), dtype=np.float64)

for col_idx in range(noise_samples):
    
    counts, bin_ranges = np.histogram(radial_strat[:,col_idx], bins, range=(0,1.0))
    
    radial_frequencies[:,col_idx] = 100 * counts / np.sum(counts)
    
plt.plot(w_eff_linspace, radial_frequencies[:,-1])
    
# shift_counts

configs_per = np.shape(shift_strat)[0]
noise_samples = np.shape(shift_strat)[1]

bins = 70

noise_linspace = np.linspace(0, 20, noise_samples)
w_eff_linspace = np.linspace(0,1, bins)

noise_meshgrid, w_eff_meshgrid = np.meshgrid(noise_linspace, w_eff_linspace)

shift_frequencies = np.ndarray((bins,noise_samples), dtype=np.float64)

for col_idx in range(noise_samples):
    
    counts, bin_ranges = np.histogram(shift_strat[:,col_idx], bins, range=(0,1.0))
    
    shift_frequencies[:,col_idx] = 100 * counts / np.sum(counts)
    
plt.plot(w_eff_linspace, shift_frequencies[:,-1])
plt.savefig(os.path.join(path,'test.png'), dpi=400, bbox_inches='tight')