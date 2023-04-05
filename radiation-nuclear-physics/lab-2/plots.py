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

def calibrationIsotopes() -> None:
    
    

def backgroundRadiation() -> None:
    
    # load up the data
    
    bin_counts, mca_coefficients = loadAsciiSpectrum(os.path.join('data','calibration_spectrum_bg.Spe'))
    
    energy_counts = bin_counts.astype(dtype=np.float64)
    
    # map bin numbers to gamma ray energies
    
    energy_counts[:,0] = mca_coefficients[0] + mca_coefficients[1] * energy_counts[:,0] + mca_coefficients[2] * energy_counts[:,0] ** 2
    smooth_energy_counts = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 6)
    
    # set limits for plotting
    
    xlim = (0,1750)
    ylim = (0,50)
    aspect = 1.5
    aspect = (xlim[1]-xlim[0]) / (aspect * (ylim[1]-ylim[0]))
    
    # plot raw values
    
    plt.plot(energy_counts[:,0], energy_counts[:,1], 'b', linewidth=0.75)
    plt.xlabel(r'\bf{Gamma Ray Energery (keV)}',fontsize=16)
    plt.xticks(fontsize=14)
    plt.xlim(xlim)
    plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
    plt.ylim(ylim)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.gca().set_aspect(aspect)
    plt.savefig('figures/background_counts_raw.png', dpi=400, bbox_inches='tight')
    plt.close()
    
    # plot smoothed values
    
    plt.plot(energy_counts[:,0], smooth_energy_counts, 'k', linewidth=0.75)
    plt.xlabel(r'\bf{Gamma Ray Energery (keV)}',fontsize=16)
    plt.xticks(fontsize=14)
    plt.xlim(xlim)
    plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
    plt.ylim(ylim)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.gca().set_aspect(aspect)
    plt.savefig('figures/background_counts_smooth.png', dpi=400, bbox_inches='tight')
    plt.close()
    
    # plot smoothed values atop raw ones
    
    plt.plot(energy_counts[:,0], energy_counts[:,1], c='b', alpha=0.5, linewidth=0.5, label=r'\bf{Raw Counts}')
    plt.plot(energy_counts[:,0], smooth_energy_counts, 'k', linewidth=1.0, label=r'$\sigma=3$ \bf{Gaussian Smooth}')
    plt.legend(prop={'size':14})
    plt.xlabel(r'\bf{Gamma Ray Energery (keV)}',fontsize=16)
    plt.xticks(fontsize=14)
    plt.xlim(xlim)
    plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
    plt.ylim(ylim)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.gca().set_aspect(aspect)
    plt.savefig('figures/background_counts_overlay.png', dpi=400, bbox_inches='tight')
    plt.close()
    
def mysteryIsotope() -> None:
    
    # load up the background data
    
    bg_bin_counts, bg_mca_coefficients = loadAsciiSpectrum(os.path.join('data','calibration_spectrum_bg.Spe'))
    
    bg_energy_counts = bg_bin_counts.astype(dtype=np.float64)
    
    # map bin numbers to gamma ray energies
    
    bg_energy_counts[:,0] = bg_mca_coefficients[0] + bg_mca_coefficients[1] * bg_energy_counts[:,0] + bg_mca_coefficients[2] * bg_energy_counts[:,0] ** 2
    bg_smooth_energy_counts = scipy.ndimage.gaussian_filter1d(bg_energy_counts[:,1], 6)
    
    mystery_files = sorted(glob.glob(os.path.join('data','mystery_*.Spe')))
    
    total_counts = np.zeros((mystery_file,2), dtype=np.float64)
    
    file_counter = 0
    
    for mystery_file in mystery_files:
        
        # load up the data
    
        bin_counts, mca_coefficients = loadAsciiSpectrum(mystery_file)
        
        mystery_file = os.path.basename(mystery_file)
        mystery_file = os.path.splitext(mystery_file)[0]
        
        print(mystery_file)
        
        energy_counts = bin_counts.astype(dtype=np.float64)
        
        total_counts[file_counter,1] = np.sum(bin_counts[:,1])
        
        # map bin numbers to gamma ray energies
        
        energy_counts[:,0] = mca_coefficients[0] + mca_coefficients[1] * energy_counts[:,0] + mca_coefficients[2] * energy_counts[:,0] ** 2
        smooth_energy_counts = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 6)
        
        # set limits for plotting
        
        xlim = (0,1750)
        ylim = (0,200)
        aspect = 1.5
        aspect = (xlim[1]-xlim[0]) / (aspect * (ylim[1]-ylim[0]))
        
        # plot raw values
        
        plt.plot(bg_energy_counts[:,0], bg_smooth_energy_counts, 'k--', alpha=0.5, linewidth=0.75, label=r'$\sigma=3$ \bf{Background}')
        plt.plot(energy_counts[:,0], energy_counts[:,1], 'b', linewidth=0.75, label=r'\bf{Raw Counts}')
        plt.legend(prop={'size':14})
        plt.xlabel(r'\bf{Gamma Ray Energery (keV)}',fontsize=16)
        plt.xticks(fontsize=14)
        plt.xlim(xlim)
        plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
        plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.gca().set_aspect(aspect)
        plt.savefig(f'figures/{mystery_file}_background_counts_raw.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        # plot smoothed values
        
        plt.plot(bg_energy_counts[:,0], bg_smooth_energy_counts, 'k--', alpha=0.5, linewidth=0.75, label=r'$\sigma=3$ \bf{Background}')
        plt.plot(energy_counts[:,0], smooth_energy_counts, 'k', linewidth=0.75, label=r'$\sigma=3$ \bf{Gaussian Smooth}')
        plt.legend(prop={'size':14})
        plt.xlabel(r'\bf{Gamma Ray Energery (keV)}',fontsize=16)
        plt.xticks(fontsize=14)
        plt.xlim(xlim)
        plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
        plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.gca().set_aspect(aspect)
        plt.savefig(f'figures/{mystery_file}_background_counts_smooth.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        # plot smoothed values atop raw ones
        
        plt.plot(bg_energy_counts[:,0], bg_smooth_energy_counts, 'k--', alpha=0.5, linewidth=0.75, label=r'$\sigma=3$ \bf{Background}')
        plt.plot(energy_counts[:,0], energy_counts[:,1], c='b', alpha=0.5, linewidth=0.5, label=r'\bf{Raw Counts}')
        plt.plot(energy_counts[:,0], smooth_energy_counts, 'k', linewidth=1.0, label=r'$\sigma=3$ \bf{Gaussian Smooth}')
        plt.legend(prop={'size':14})
        plt.xlabel(r'\bf{Gamma Ray Energery (keV)}',fontsize=16)
        plt.xticks(fontsize=14)
        plt.xlim(xlim)
        plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
        plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.gca().set_aspect(aspect)
        plt.savefig(f'figures/{mystery_file}_background_counts_overlay.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        file_counter += 1
        
    # load up the data
    
    baseline_bin_counts, baseline_mca_coefficients = loadAsciiSpectrum(mystery_files[0])
    
    baseline_energy_counts = baseline_bin_counts.astype(dtype=np.float64)
    
    # map bin numbers to gamma ray energies
    
    baseline_energy_counts[:,0] = baseline_mca_coefficients[0] + baseline_mca_coefficients[1] * baseline_energy_counts[:,0] + baseline_mca_coefficients[2] * baseline_energy_counts[:,0] ** 2
    baseline_smooth_energy_counts = scipy.ndimage.gaussian_filter1d(baseline_energy_counts[:,1], 6)
    
    # set limits for plotting
        
    xlim = (0,1750)
    ylim = (-50,5)
    aspect = 1.5
    aspect = (xlim[1]-xlim[0]) / (aspect * (ylim[1]-ylim[0]))
    
    counter = 1
    
    plt.plot(list(xlim), [0,0], 'k', linewidth=1.0, label=r'\bf{First Recording}')
        
    for mystery_file in mystery_files[1:]:
        
        # load up the data
    
        bin_counts, mca_coefficients = loadAsciiSpectrum(mystery_file)
        
        mystery_file = os.path.basename(mystery_file)
        mystery_file = os.path.splitext(mystery_file)[0]
        
        energy_counts = bin_counts.astype(dtype=np.float64)
        
        # map bin numbers to gamma ray energies
        
        energy_counts[:,0] = mca_coefficients[0] + mca_coefficients[1] * energy_counts[:,0] + mca_coefficients[2] * energy_counts[:,0] ** 2
        smooth_energy_counts = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 6)
        
        # plot smoothed values
        
        plt.plot(energy_counts[:,0], smooth_energy_counts - baseline_smooth_energy_counts, 'k', linewidth=0.75, alpha=0.9 - 0.4 * (counter / (len(mystery_files) - 1)))
        
        counter += 1
        
    plt.legend(prop={'size':14})
    plt.xlabel(r'\bf{Gamma Ray Energery (keV)}',fontsize=16)
    plt.xticks(fontsize=14)
    plt.xlim(xlim)
    plt.ylabel(r'$\Delta$ \bf{Detections from First Recording}', fontsize=16)
    plt.ylim(ylim)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.gca().set_aspect(aspect)
    plt.savefig(f'figures/difference_counts_smooth.png', dpi=400, bbox_inches='tight')
    plt.close()

def main() -> None:
    
    calibrationIsotopes()
    backgroundRadiation()
    mysteryIsotope()

if __name__ == '__main__':
    
    main()