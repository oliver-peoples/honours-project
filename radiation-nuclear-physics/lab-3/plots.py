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
    
    calibration_isotopes = sorted(glob.glob(os.path.join('data', 'calibration_spectrum_*.Spe')))
    
    # set limits for plotting
        
    xlim = (0,1750)
    ylim = (0,5000)
    aspect = 1.75
    aspect = (xlim[1]-xlim[0]) / (aspect * (ylim[1]-ylim[0]))
    
    _, truth_mca_coefficients, _ = loadAsciiSpectrum(os.path.join('data','calibration_spectrum_bg.Spe'))
    
    # final calibration atop original calibration
    
    for calibration_isotope in calibration_isotopes:
        
        # load up the data
    
        bin_counts, mca_coefficients, _ = loadAsciiSpectrum(calibration_isotope)
        
        calibration_isotope = os.path.basename(calibration_isotope)
        calibration_isotope = os.path.splitext(calibration_isotope)[0]
        
        print(calibration_isotope)
        
        energy_counts = bin_counts.astype(dtype=np.float64)
        
        # map bin numbers to gamma ray energies
        
        energy_counts_og_mca = mca_coefficients[0] + mca_coefficients[1] * energy_counts[:,0] + mca_coefficients[2] * energy_counts[:,0] ** 2
        energy_counts_final_mca = truth_mca_coefficients[0] + truth_mca_coefficients[1] * energy_counts[:,0] + truth_mca_coefficients[2] * energy_counts[:,0] ** 2
        
        smooth_energy_counts_og_mca = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 6)
        smooth_energy_counts_final_mca = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 6)
        
        # plot smoothed values atop raw ones
        
        plt.plot(energy_counts_og_mca, energy_counts[:,1], c='b', alpha=0.5, linewidth=0.5, label=r'\bf{Raw Spectra C.A.R.}')
        plt.plot(energy_counts_og_mca, smooth_energy_counts_og_mca, 'k', linewidth=0.75, label=r'$\sigma_{G}=3$ \bf{Spectra C.A.R.}')
        plt.plot(energy_counts_final_mca, smooth_energy_counts_final_mca, 'k--', alpha=0.8, linewidth=0.75, label=r'$\sigma_{G}=3$ \bf{Spectra F.C.}')
        plt.legend(prop={'size':14})
        plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
        plt.xticks(fontsize=14)
        plt.xlim(xlim)
        plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
        plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.gca().set_aspect(aspect)
        plt.savefig(f'figures/{calibration_isotope}_counts_overlay_both_mca.png', dpi=400, bbox_inches='tight')
        plt.close()   
    
    # original calibration
    
    for calibration_isotope in calibration_isotopes:
        
        # load up the data
    
        bin_counts, mca_coefficients, _ = loadAsciiSpectrum(calibration_isotope)
        
        calibration_isotope = os.path.basename(calibration_isotope)
        calibration_isotope = os.path.splitext(calibration_isotope)[0]
        
        print(calibration_isotope)
        
        energy_counts = bin_counts.astype(dtype=np.float64)
        
        # map bin numbers to gamma ray energies
        
        energy_counts[:,0] = mca_coefficients[0] + mca_coefficients[1] * energy_counts[:,0] + mca_coefficients[2] * energy_counts[:,0] ** 2
        smooth_energy_counts = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 6)
        
        # plot raw values
        
        plt.plot(energy_counts[:,0], energy_counts[:,1], 'b', linewidth=0.75, label=r'\bf{Raw Spectra}')
        plt.legend(prop={'size':14})
        plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
        plt.xticks(fontsize=14)
        plt.xlim(xlim)
        plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
        plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.gca().set_aspect(aspect)
        plt.savefig(f'figures/{calibration_isotope}_counts_raw_og_mca.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        # plot smoothed values
        
        plt.plot(energy_counts[:,0], smooth_energy_counts, 'k', linewidth=0.75, label=r'$\sigma_{G}=3$ \bf{Spectra}')
        plt.legend(prop={'size':14})
        plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
        plt.xticks(fontsize=14)
        plt.xlim(xlim)
        plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
        plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.gca().set_aspect(aspect)
        plt.savefig(f'figures/{calibration_isotope}_counts_smooth_og_mca.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        # plot smoothed values atop raw ones
        
        plt.plot(energy_counts[:,0], energy_counts[:,1], c='b', alpha=0.5, linewidth=0.5, label=r'\bf{Raw Spectra}')
        plt.plot(energy_counts[:,0], smooth_energy_counts, 'k', linewidth=1.0, label=r'$\sigma_{G}=3$ \bf{Spectra}')
        plt.legend(prop={'size':14})
        plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
        plt.xticks(fontsize=14)
        plt.xlim(xlim)
        plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
        plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.gca().set_aspect(aspect)
        plt.savefig(f'figures/{calibration_isotope}_counts_overlay_og_mca.png', dpi=400, bbox_inches='tight')
        plt.close()
        
    # final calibration
    
    for calibration_isotope in calibration_isotopes:
        
        # load up the data
    
        bin_counts, mca_coefficients, _ = loadAsciiSpectrum(calibration_isotope)
        
        calibration_isotope = os.path.basename(calibration_isotope)
        calibration_isotope = os.path.splitext(calibration_isotope)[0]
        
        print(calibration_isotope)
        
        energy_counts = bin_counts.astype(dtype=np.float64)
        
        # map bin numbers to gamma ray energies
        
        energy_counts[:,0] = truth_mca_coefficients[0] + truth_mca_coefficients[1] * energy_counts[:,0] + truth_mca_coefficients[2] * energy_counts[:,0] ** 2
        smooth_energy_counts = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 6)
        
        # plot raw values
        
        plt.plot(energy_counts[:,0], energy_counts[:,1], 'b', linewidth=0.75, label=r'\bf{Raw Spectra}')
        plt.legend(prop={'size':14})
        plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
        plt.xticks(fontsize=14)
        plt.xlim(xlim)
        plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
        plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.gca().set_aspect(aspect)
        plt.savefig(f'figures/{calibration_isotope}_counts_raw_final_mca.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        # plot smoothed values
        
        plt.plot(energy_counts[:,0], smooth_energy_counts, 'k', linewidth=0.75, label=r'$\sigma_{G}=3$ \bf{Spectra}')
        plt.legend(prop={'size':14})
        plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
        plt.xticks(fontsize=14)
        plt.xlim(xlim)
        plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
        plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.gca().set_aspect(aspect)
        plt.savefig(f'figures/{calibration_isotope}_counts_smooth_final_mca.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        # plot smoothed values atop raw ones
        
        plt.plot(energy_counts[:,0], energy_counts[:,1], c='b', alpha=0.5, linewidth=0.5, label=r'\bf{Raw Spectra}')
        plt.plot(energy_counts[:,0], smooth_energy_counts, 'k', linewidth=1.0, label=r'$\sigma_{G}=3$ \bf{Spectra}')
        plt.legend(prop={'size':14})
        plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
        plt.xticks(fontsize=14)
        plt.xlim(xlim)
        plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
        plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.gca().set_aspect(aspect)
        plt.savefig(f'figures/{calibration_isotope}_counts_overlay_final_mca.png', dpi=400, bbox_inches='tight')
        plt.close()   
    

def backgroundRadiation() -> None:
    
    # load up the data
    
    bin_counts, mca_coefficients, _ = loadAsciiSpectrum(os.path.join('data','calibration_spectrum_bg.Spe'))
    
    energy_counts = bin_counts.astype(dtype=np.float64)
    
    # map bin numbers to gamma ray energies
    
    energy_counts[:,0] = mca_coefficients[0] + mca_coefficients[1] * energy_counts[:,0] + mca_coefficients[2] * energy_counts[:,0] ** 2
    smooth_energy_counts = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 6)
    
    sum_counts_lt = np.sum(energy_counts[np.where(energy_counts[:,0] < 400),1])
    print(400, sum_counts_lt / np.sum(energy_counts[:,1]))
    sum_counts_lt = np.sum(energy_counts[np.where(energy_counts[:,0] < 200),1])
    print(200, sum_counts_lt / np.sum(energy_counts[:,1]))
    # set limits for plotting
    
    xlim = (0,1750)
    ylim = (0,50)
    aspect = 1.75
    aspect = (xlim[1]-xlim[0]) / (aspect * (ylim[1]-ylim[0]))
    
    # plot raw values
    
    plt.plot(energy_counts[:,0], energy_counts[:,1], 'b', linewidth=0.75)
    plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
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
    plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
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
    
    plt.plot(energy_counts[:,0], energy_counts[:,1], c='b', alpha=0.5, linewidth=0.5, label=r'\bf{Raw Spectra}')
    plt.plot(energy_counts[:,0], smooth_energy_counts, 'k', linewidth=1.0, label=r'$\sigma_{G}=3$ \bf{Spectra}')
    plt.legend(prop={'size':14})
    plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
    plt.xticks(fontsize=14)
    plt.xlim(xlim)
    plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
    plt.ylim(ylim)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.gca().set_aspect(aspect)
    plt.savefig('figures/background_counts_overlay.png', dpi=400, bbox_inches='tight')
    plt.close()

def main() -> None:
    
    calibrationIsotopes()
    backgroundRadiation()

if __name__ == '__main__':
    
    main()