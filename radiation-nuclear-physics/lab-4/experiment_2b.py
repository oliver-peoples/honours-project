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

xlim = [calibration_mca_coefficients[0],0]
ylim = [0,0]

shoulder_start = 1275.
shoulder_end = 1550.

roi = np.where(np.logical_and(bg_spectra[:,0] > shoulder_start,bg_spectra[:,0] < shoulder_end))[0]
left_roi = np.where(bg_spectra[:,0] < shoulder_start)[0]
right_roi = np.where(bg_spectra[:,0] > shoulder_end)[0]
    
energy_counts, _, epoch, integration_time = loadAsciiSpectrum('raw-data/V_2B.Spe')

energy_counts = np.asarray(energy_counts, dtype=np.float64)

energy_counts[:,0] = calibration_mca_coefficients[0] + calibration_mca_coefficients[1] * energy_counts[:,0] + calibration_mca_coefficients[2] * energy_counts[:,0] ** 2
energy_counts[:,1] = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 3)
energy_counts[:,1] -= bg_spectra[:,1] * integration_time

roi_vals = scipy.ndimage.gaussian_filter1d(energy_counts[roi,1], 11)
peaks = scipy.signal.find_peaks(roi_vals, 12)
print(peaks)
peak_energy = energy_counts[roi,0][peaks[0][0]]

plt.plot(energy_counts[roi,0], energy_counts[roi,1], c='r', linewidth=0.75)
plt.plot(energy_counts[left_roi,0], energy_counts[left_roi,1], c='b', linewidth=0.75)
plt.plot(energy_counts[right_roi,0], energy_counts[right_roi,1], c='b', linewidth=0.75)
xlim = [xlim[0],max(xlim[1],np.max(energy_counts[:,0]))]
ylim = [0,max(ylim[1],np.max(energy_counts[:,1]))]
    
ylim[1] = 10 * np.ceil(ylim[1] / 10)

plt.plot([peak_energy,peak_energy],ylim,'r--', linewidth=0.75, label=r'$' + f'{peak_energy:4.1f}' + r'$ $\mathrm{keV}$')
aspect = (xlim[1]-xlim[0]) / (aspect_ratio * (ylim[1]-ylim[0]))    
plt.legend(prop={'size':14})
plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
plt.xticks(fontsize=14)
plt.xlim(xlim)
plt.ylabel(r'\bf{\# of Counts}', fontsize=16)
plt.ylim(ylim)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.gca().set_aspect(aspect)
plt.savefig(f'figures/decay_2b.png', dpi=400, bbox_inches='tight')
plt.close()