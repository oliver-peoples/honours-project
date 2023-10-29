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

# radial_counts

# radial_strat = np.genfromtxt(os.path.join(path,'data','w_eff_bar_2_19_1,3,5,7,11,15.csv'), delimiter=',', skip_header=1)
radial_strat = np.genfromtxt(os.path.join(path,'data','chi2_w_eff_bar_normed.csv'), delimiter=',', skip_header=1)

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

lt_thresh = np.where(w_eff_linspace < 0.05)[0]
radial_frequencies_thresh = radial_frequencies[lt_thresh,:]
print(radial_frequencies_thresh[:,0])
radial_frequencies_normed = radial_frequencies[:,-1]
# radial_frequencies_normed /= np.max(radial_frequencies_normed)
   
# plt.step(w_eff_linspace, radial_frequencies_normed, c='cyan', linestyle='--')
    
# shift_counts

# shift_strat = np.genfromtxt(os.path.join(path,'data','w_eff_bar_2_19_1,3,5,9,13,17.csv'), delimiter=',', skip_header=1)
shift_strat = np.genfromtxt(os.path.join(path,'data','chi2_w_eff_bar_worboy.csv'), delimiter=',', skip_header=1)

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

lt_thresh = np.where(w_eff_linspace < 0.05)[0]
shift_frequencies_thresh = shift_frequencies[lt_thresh,:]
print(shift_frequencies_thresh[:,0])
shift_frequencies_normed = shift_frequencies[:,-1]
# shift_frequencies_normed /= np.max(shift_frequencies_normed)

# plt.step(w_eff_linspace, shift_frequencies_normed, c='magenta', linestyle='--')

# fine_radial_counts

fine_radial_strat = np.genfromtxt(os.path.join(path,'data','w_eff_bar_2_19_1,3,5,7,11,15_fine.csv'), delimiter=',', skip_header=1)

configs_per = np.shape(fine_radial_strat)[0]
noise_samples = np.shape(fine_radial_strat)[1]

bins = 70

noise_linspace = np.linspace(0, 20, noise_samples)
w_eff_linspace = np.linspace(0,1, bins)

noise_meshgrid, w_eff_meshgrid = np.meshgrid(noise_linspace, w_eff_linspace)

fine_radial_frequencies = np.ndarray((bins,noise_samples), dtype=np.float64)

for col_idx in range(noise_samples):
    
    counts, bin_ranges = np.histogram(fine_radial_strat[:,col_idx], bins, range=(0,1.0))
    
    fine_radial_frequencies[:,col_idx] = 100 * counts / np.sum(counts)
    
fine_radial_frequencies_normed = fine_radial_frequencies[:,-1]
# fine_radial_frequencies_normed /= np.max(fine_radial_frequencies_normed)
    
plt.plot(w_eff_linspace, fine_radial_frequencies_normed, c='cyan', label=r'$\mathrm{Radial\;Method}$')

# fine_shift_counts

fine_shift_strat = np.genfromtxt(os.path.join(path,'data','w_eff_bar_2_19_1,3,5,9,13,17_fine.csv'), delimiter=',', skip_header=1)

configs_per = np.shape(fine_shift_strat)[0]
noise_samples = np.shape(fine_shift_strat)[1]

bins = 70

noise_linspace = np.linspace(0, 20, noise_samples)
w_eff_linspace = np.linspace(0,1, bins)

noise_meshgrid, w_eff_meshgrid = np.meshgrid(noise_linspace, w_eff_linspace)

fine_shift_frequencies = np.ndarray((bins,noise_samples), dtype=np.float64)

for col_idx in range(noise_samples):
    
    counts, bin_ranges = np.histogram(fine_shift_strat[:,col_idx], bins, range=(0,1.0))
    
    fine_shift_frequencies[:,col_idx] = 100 * counts / np.sum(counts)

lt_thresh = np.where(w_eff_linspace < 0.05)[0]
fine_shift_frequencies_thresh = fine_shift_frequencies[lt_thresh,:]
print(fine_shift_frequencies_thresh[:,0])
fine_shift_frequencies_normed = fine_shift_frequencies[:,-1]
# fine_shift_frequencies_normed /= np.max(fine_shift_frequencies_normed)
    
plt.plot(w_eff_linspace, fine_shift_frequencies_normed, c='magenta', label=r'$\mathrm{Rotated\;Method}$')

plt.legend(fontsize=10)
plt.xlabel(r'$\bar{w}_{\mathrm{eff}}$', fontsize=10)
plt.xlim(0,1)
plt.xticks(fontsize=10)
plt.ylabel(r'$\mathrm{Relative\;Frequency}$', fontsize=10)
plt.ylim(0,15)
plt.yticks(fontsize=10)
plt.gca().set_aspect(1/(2.5 * 15))
plt.gcf().set_figwidth(val=0.99 * 15.3978 * cm)
plt.savefig(os.path.join(path,'g2_ring_comparison.png'), dpi=500, bbox_inches='tight')