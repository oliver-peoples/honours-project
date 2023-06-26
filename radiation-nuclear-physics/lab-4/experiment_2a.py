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

matplotlib.rcParams['text.usetex'] = True

aspect_ratio = 1.75

calibration_mca_coefficients = np.genfromtxt('calibration_mca_coefficients.csv', delimiter=',')

# get background

bg_spectra, _, _, bg_integration_time = loadAsciiSpectrum('raw-data/bg.Spe')

bg_spectra = np.asarray(bg_spectra, dtype=np.float64)

bg_spectra[:,0] = calibration_mca_coefficients[0] + calibration_mca_coefficients[1] * bg_spectra[:,0] + calibration_mca_coefficients[2] * bg_spectra[:,0] ** 2
bg_spectra[:,1] = scipy.ndimage.gaussian_filter1d(bg_spectra[:,1], 3) / bg_integration_time

#  get spectra from experiment

spectra_files = sorted(glob.glob('raw-data/DECAY_2A_*.Spe'))

num_spectra_files = len(spectra_files)

xlim = [calibration_mca_coefficients[0],0]
ylim = [0,0]

shoulder_start = 750.
shoulder_end = 1000.

roi = np.where(np.logical_and(bg_spectra[:,0] > shoulder_start,bg_spectra[:,0] < shoulder_end))[0]
left_roi = np.where(bg_spectra[:,0] < shoulder_start)[0]
right_roi = np.where(bg_spectra[:,0] > shoulder_end)[0]

totals = np.zeros((num_spectra_files,2), dtype=np.float64)

for spectra_file, idx in zip(spectra_files, range(num_spectra_files)):
    
    energy_counts, _, epoch, integration_time = loadAsciiSpectrum(spectra_file)
    
    energy_counts = np.asarray(energy_counts, dtype=np.float64)
    
    energy_counts[:,0] = calibration_mca_coefficients[0] + calibration_mca_coefficients[1] * energy_counts[:,0] + calibration_mca_coefficients[2] * energy_counts[:,0] ** 2
    energy_counts[:,1] = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 3)
    energy_counts[:,1] -= bg_spectra[:,1] * integration_time

    plt.plot(energy_counts[roi,0], energy_counts[roi,1], c='r', linewidth=0.75, alpha = 0.6 - 0.4 * (float(idx) / float(num_spectra_files - 1)))
    plt.plot(energy_counts[left_roi,0], energy_counts[left_roi,1], c='b', linewidth=0.75, alpha = 0.6 - 0.4 * (float(idx) / float(num_spectra_files - 1)))
    plt.plot(energy_counts[right_roi,0], energy_counts[right_roi,1], c='b', linewidth=0.75, alpha = 0.6 - 0.4 * (float(idx) / float(num_spectra_files - 1)))
    
    totals[idx,0] = epoch
    totals[idx,1] = np.sum(energy_counts[roi,1])
    
    xlim = [xlim[0],max(xlim[1],np.max(energy_counts[:,0]))]
    ylim = [0,max(ylim[1],np.max(energy_counts[:,1]))]
    
ylim[1] = 10 * np.ceil(ylim[1] / 10)

aspect = (xlim[1]-xlim[0]) / (aspect_ratio * (ylim[1]-ylim[0]))    
# plt.legend(prop={'size':14})
plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
plt.xticks(fontsize=14)
plt.xlim(xlim)
plt.ylabel(r'\bf{\# of Counts}', fontsize=16)
plt.ylim(ylim)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.gca().set_aspect(aspect)
plt.savefig(f'figures/decay_2a.png', dpi=400, bbox_inches='tight')
plt.close()

totals[:,0] -= totals[0,0]
totals[:,1] -= totals[-1,1]
totals[:,1] += 0.1

result = scipy.stats.linregress(totals[:,0],np.log(totals[:,1] - np.log(totals[0,1])))

totals_ideal = np.zeros((1000,2), dtype=np.float64)
totals_ideal[:,0] = np.linspace(0,np.max(totals[:,0]),1000)

totals_ideal[:,1] = totals[0,1] * np.exp(result.slope * totals_ideal[:,0])

xlim = [0,np.max(totals[:,0])]
ylim = [np.min(totals[:,1]) - 100,np.max(totals[:,1]) + 100]

aspect = (xlim[1]-xlim[0]) / (aspect_ratio * (ylim[1]-ylim[0]))  

plt.plot(totals[:,0],totals[:,1], c='b', linewidth=0.75)
plt.plot(totals_ideal[:,0],totals_ideal[:,1], 'k-.', linewidth=0.75, label=r'$\lambda=' + f'{-1. * result.slope:1.4f}' + r'$ \bf{Fit}')

plt.xlabel(r'$T+$ \bf{(minutes)}',fontsize=16)
plt.xticks(fontsize=14)
plt.xlim(xlim)
plt.ylabel(r'\bf{ROI Sum of Counts}', fontsize=16)
plt.ylim(ylim)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.gca().set_aspect(aspect)
plt.savefig(f'figures/decay_2a_roi.png', dpi=400, bbox_inches='tight')
plt.close()