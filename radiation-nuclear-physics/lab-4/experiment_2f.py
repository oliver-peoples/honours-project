# Oliver Kirkpatrick - oliver-peoples - 10.133.8.27 - finn@finna - Razer Blade Stealth 15
# CONDA_ENV - Python 3.10.9 - Balloon Artificial Research Environment for Navigation and Autonomy (BallARENA) TASDCRC
# 5/4/23 17:09

from loadAsciiSpectrum import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import glob
import scipy.signal
import scipy.ndimage
from datetime import datetime
import calendar
import scipy.stats

matplotlib.rcParams['text.usetex'] = True

aspect_ratio = 1.75

calibration_mca_coefficients = np.genfromtxt('calibration_mca_coefficients.csv', delimiter=',')

# get background

bg_spectra, _, _, bg_integration_time = loadAsciiSpectrum('raw-data/bg.Spe')

bg_spectra = np.asarray(bg_spectra, dtype=np.float64)

bg_spectra[:,0] = calibration_mca_coefficients[0] + calibration_mca_coefficients[1] * bg_spectra[:,0] + calibration_mca_coefficients[2] * bg_spectra[:,0] ** 2
bg_spectra[:,1] = scipy.ndimage.gaussian_filter1d(bg_spectra[:,1], 3) / bg_integration_time

#  get spectra from experiment

spectra_files = sorted(glob.glob('raw-data/DECAY_2F_*.Spe'))

num_spectra_files = len(spectra_files)

xlim = [calibration_mca_coefficients[0],0]
ylim = [0,0]

center_val_1 = 844
shoulder_start_1 = center_val_1 - 60
shoulder_end_1 = center_val_1 + 60

roi_1 = np.where(np.logical_and(bg_spectra[:,0] > shoulder_start_1,bg_spectra[:,0] < shoulder_end_1))[0]
roi_1 = [roi_1[0] - 1,*roi_1,roi_1[-1] + 1]
left_roi = np.where(bg_spectra[:,0] < shoulder_start_1)[0]
# right_roi_1 = np.where(bg_spectra[:,0] > shoulder_end_1)[0]

center_val_2 = 1015
shoulder_start_2 = center_val_2 - 60
shoulder_end_2 = center_val_2 + 60

roi_2 = np.where(np.logical_and(bg_spectra[:,0] > shoulder_start_2,bg_spectra[:,0] < shoulder_end_2))[0]
roi_2 = [roi_2[0] - 1,*roi_2,roi_2[-1] + 1]
# left_roi_2 = np.where(bg_spectra[:,0] < shoulder_start_2)[0]
right_roi = np.where(bg_spectra[:,0] > shoulder_end_2)[0]

center_roi = np.where(np.logical_and(bg_spectra[:,0] > shoulder_end_1,bg_spectra[:,0] < shoulder_start_2))[0]

totals = np.zeros((num_spectra_files,3), dtype=np.float64)

for spectra_file, idx in zip(spectra_files, range(num_spectra_files)):
    
    energy_counts, _, epoch, integration_time = loadAsciiSpectrum(spectra_file)
    
    energy_counts = np.asarray(energy_counts, dtype=np.float64)
    
    energy_counts[:,0] = calibration_mca_coefficients[0] + calibration_mca_coefficients[1] * energy_counts[:,0] + calibration_mca_coefficients[2] * energy_counts[:,0] ** 2
    energy_counts[:,1] = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 3)
    energy_counts[:,1] -= bg_spectra[:,1] * integration_time

    plt.plot(energy_counts[roi_1,0], energy_counts[roi_1,1], c='r', linewidth=0.75, alpha = 0.6 - 0.4 * (float(idx) / float(num_spectra_files - 1)))
    plt.plot(energy_counts[roi_2,0], energy_counts[roi_2,1], color='magenta', linewidth=0.75, alpha = 0.6 - 0.4 * (float(idx) / float(num_spectra_files - 1)))
    
    plt.plot(energy_counts[left_roi,0], energy_counts[left_roi,1], c='b', linewidth=0.75, alpha = 0.6 - 0.4 * (float(idx) / float(num_spectra_files - 1)))
    plt.plot(energy_counts[center_roi,0], energy_counts[center_roi,1], c='b', linewidth=0.75, alpha = 0.6 - 0.4 * (float(idx) / float(num_spectra_files - 1)))
    plt.plot(energy_counts[right_roi,0], energy_counts[right_roi,1], c='b', linewidth=0.75, alpha = 0.6 - 0.4 * (float(idx) / float(num_spectra_files - 1)))
    
    totals[idx,0] = epoch
    totals[idx,1] = np.sum(energy_counts[roi_1,1])
    totals[idx,2] = np.sum(energy_counts[roi_2,1])
    
    xlim = [xlim[0],max(xlim[1],np.max(energy_counts[:,0]))]
    ylim = [0,max(ylim[1],np.max(energy_counts[:,1]))]
    
ylim[1] = 10 * np.ceil(ylim[1] / 10)

total_peaks = totals[:,1] + totals[:,2]
print(np.mean(totals[:,1] / total_peaks))
print(np.mean(totals[:,2] / total_peaks))

aspect = (xlim[1]-xlim[0]) / (aspect_ratio * (ylim[1]-ylim[0]))    

plt.plot([],[], c='r', linewidth=0.75, alpha = 0.6, label=r'$844\;\mathrm{keV\;Peak}$')
plt.plot([],[], color='magenta', linewidth=0.75, alpha = 0.6, label=r'$1015\;\mathrm{keV\;Peak}$')

plt.legend(prop={'size':14})
plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
plt.xticks(fontsize=14)
plt.xlim(xlim)
plt.ylabel(r'\bf{\# of Counts}', fontsize=16)
plt.ylim(ylim)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.gca().set_aspect(aspect)
plt.savefig(f'figures/decay_2f.png', dpi=400, bbox_inches='tight')
plt.close()

totals[:,0] -= totals[0,0]

result = scipy.stats.linregress(totals[:,0],np.log(totals[:,1] - np.log(totals[0,1])))

totals_ideal = np.zeros((1000,2), dtype=np.float64)
totals_ideal[:,0] = np.linspace(0,np.max(totals[:,0]),1000)

totals_ideal[:,1] = totals[0,1] * np.exp(result.slope * totals_ideal[:,0])

print(np.log(0.5)/result.slope)

xlim = [0,np.max(totals[:,0])]
ylim = [0,np.max(totals[:,1]) + 100]

aspect = (xlim[1]-xlim[0]) / (aspect_ratio * (ylim[1]-ylim[0]))  

plt.plot(totals[:,0],totals[:,1], c='b', linewidth=0.75, label=r'\bf{ROI Counts}')
plt.plot(totals_ideal[:,0],totals_ideal[:,1], 'k--', linewidth=0.75, label=r'$\lambda=' + f'{-1. * result.slope:1.4f}' + r'$ \bf{Fit}')
plt.legend(fontsize=16)
plt.xlabel(r'$T+$ \bf{(minutes)}',fontsize=16)
plt.xticks(fontsize=14)
plt.xlim(xlim)
plt.ylabel(r'\bf{ROI Sum of Counts}', fontsize=16)
plt.ylim(ylim)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.gca().set_aspect(aspect)
plt.savefig(f'figures/decay_2f_roi.png', dpi=400, bbox_inches='tight')
plt.close()