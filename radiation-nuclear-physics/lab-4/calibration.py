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

isotopes = { '60Co':r'$^{60}\mathrm{Co}$','137Cs':r'$^{137}\mathrm{Cs}$','241Am':r'$^{241}\mathrm{Am}$' }

colors = [ 'r','g','b' ]

calibration_files = glob.glob('calibration/*_cal.Spe')

calibration = [0,0,0]

for calibration_file in calibration_files:
    
    _, mca_coefficients, epoch, integration_time = loadAsciiSpectrum(calibration_file)
    
    if epoch > calibration[1]:
        
        calibration = [mca_coefficients,epoch,integration_time]
    
calibration_mca_coefficients = calibration[0]

print(calibration_mca_coefficients)

counter = 0

xlim = [calibration_mca_coefficients[0],0]
ylim = [0,0]

for calibration_file in calibration_files:
    
    isotope_name = isotopes[list(isotopes.keys())[[i for i, s in enumerate(isotopes.keys()) if s in calibration_file][0]]]
    
    energy_counts, _, _, integration_time = loadAsciiSpectrum(calibration_file)
    
    print(np.sum(energy_counts[:,1]), np.sum(energy_counts[:,1]) / integration_time)
    
    energy_counts = np.asarray(energy_counts, dtype=np.float64)
    
    energy_counts[:,0] = calibration_mca_coefficients[0] + calibration_mca_coefficients[1] * energy_counts[:,0] + calibration_mca_coefficients[2] * energy_counts[:,0] ** 2
    energy_counts[:,1] = scipy.ndimage.gaussian_filter1d(energy_counts[:,1] / 1., 3)
    
    
    
    plt.plot(energy_counts[:,0], energy_counts[:,1], c=colors[counter], linewidth=0.75, label=isotope_name)
    
    xlim = [xlim[0],max(xlim[1],np.max(energy_counts[:,0]))]
    ylim = [0,max(ylim[1],np.max(energy_counts[:,1]))]
    
    counter += 1
    
ylim[1] = 500 * np.ceil(ylim[1] / 500)

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
plt.savefig(f'figures/calibrations_overlaid.png', dpi=400, bbox_inches='tight')
plt.close()