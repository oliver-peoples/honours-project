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
    
def mysteryIsotope() -> None:
    
    # load up the background data
    
    bg_bin_counts, bg_mca_coefficients, _ = loadAsciiSpectrum(os.path.join('data','calibration_spectrum_bg.Spe'))
    
    bg_energy_counts = bg_bin_counts.astype(dtype=np.float64)
    
    # map bin numbers to gamma ray energies
    
    bg_energy_counts[:,0] = bg_mca_coefficients[0] + bg_mca_coefficients[1] * bg_energy_counts[:,0] + bg_mca_coefficients[2] * bg_energy_counts[:,0] ** 2
    bg_smooth_energy_counts = scipy.ndimage.gaussian_filter1d(bg_energy_counts[:,1], 6)
    bg_smooth_energy_counts *= 2
    
    mystery_files = sorted(glob.glob(os.path.join('data','mystery_*.Spe')))
    
    total_counts = np.zeros((len(mystery_files),2), dtype=np.float64)
    
    file_counter = 0
    
    for mystery_file in mystery_files:
        
        # load up the data
    
        bin_counts, mca_coefficients, epoch_minutes = loadAsciiSpectrum(mystery_file)
        
        mystery_file = os.path.basename(mystery_file)
        mystery_file = os.path.splitext(mystery_file)[0]
        
        print(mystery_file)
        
        energy_counts = bin_counts.astype(dtype=np.float64)
        
        total_counts[file_counter,1] = np.sum(bin_counts[:,1]) - np.sum(bg_bin_counts[:,1] * 2)
        total_counts[file_counter,0] = epoch_minutes
        
        # map bin numbers to gamma ray energies
        
        energy_counts[:,0] = mca_coefficients[0] + mca_coefficients[1] * energy_counts[:,0] + mca_coefficients[2] * energy_counts[:,0] ** 2
        smooth_energy_counts = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 6)
        
        # set limits for plotting
        
        xlim = (0,1750)
        ylim = (0,200)
        aspect = 1.75
        aspect = (xlim[1]-xlim[0]) / (aspect * (ylim[1]-ylim[0]))
        
        # plot raw values
        
        plt.plot(bg_energy_counts[:,0], bg_smooth_energy_counts, 'k--', alpha=0.5, linewidth=0.75, label=r'$\sigma_{G}=3$ \bf{Background}')
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
        plt.savefig(f'figures/{mystery_file}_background_counts_raw.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        # plot smoothed values
        
        plt.plot(bg_energy_counts[:,0], bg_smooth_energy_counts, 'k--', alpha=0.5, linewidth=0.75, label=r'$\sigma_{G}=3$ \bf{Background}')
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
        plt.savefig(f'figures/{mystery_file}_background_counts_smooth.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        # plot smoothed values atop raw ones
        
        plt.plot(bg_energy_counts[:,0], bg_smooth_energy_counts, 'k--', alpha=0.5, linewidth=0.75, label=r'$\sigma_{G}=3$ \bf{Background}')
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
        plt.savefig(f'figures/{mystery_file}_background_counts_overlay.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        file_counter += 1
        
    start_epoch = total_counts[0,0]
    total_counts[:,0] -= total_counts[0,0]
    print(total_counts)
    
    # x_range = total_counts[-1,0] - total_counts[0,0]
    y_max = 10000 * round(np.max(total_counts[:,1]) / 10000) + 5000
    aspect = 1.75
    aspect = (75 / (aspect * y_max))
    
    t_span = np.linspace(0,75,10000)
    theoretical_counts = total_counts[0,1] * np.exp(-9.46e-3 * t_span)
    
    plt.plot(total_counts[:,0], total_counts[:,1], 'k--', linewidth=1.0, label=r'\bf{Empirical}')
    plt.plot(t_span, theoretical_counts, 'k--', alpha=0.5, linewidth=0.75,label=r'\bf{Theoretical')
    plt.plot([0,75],[total_counts[0,1] * 0.5,total_counts[0,1] * 0.5], 'r--', linewidth=0.75, label=r'$I_{1/2}$')
    plt.scatter(total_counts[:,0], total_counts[:,1], c='b', s=5.0)
    plt.legend(prop={'size':14})
    plt.xlabel(r'\bf{Time Since First Recording (min)} $T+$',fontsize=16)
    plt.xticks(fontsize=14)
    plt.xlim((0,75))
    plt.ylabel(r'\bf{Total Detections}', fontsize=16)
    plt.ylim((0,y_max))
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.gca().set_aspect(aspect)
    plt.savefig(f'figures/mystery_decay.png', dpi=400, bbox_inches='tight')
    plt.close()
    
    ln_I_0 = np.log(total_counts[0,1])
    ln_I = np.log(total_counts[:,1]) - ln_I_0
    
    total_counts[:,1] = ln_I
    
    for row in total_counts[:]:
        
        print(f'{row[0]:.4f}')
    
    for row in total_counts[:]:
        
        print(f'{row[1]:.4f}')
    # plot all
    
    xlim = (0,1750)
    ylim = (0,250)
    aspect = 1.75
    aspect = (xlim[1]-xlim[0]) / (aspect * (ylim[1]-ylim[0]))
    
    plt.plot(bg_energy_counts[:,0], bg_smooth_energy_counts, 'k--', alpha=0.5, linewidth=0.75, label=r'$\sigma_{G}=3$ \bf{Background}')
    
    counter = 0
    
    for mystery_file in mystery_files:
        
        # load up the data
    
        bin_counts, mca_coefficients, epoch = loadAsciiSpectrum(mystery_file)
        
        epoch -= start_epoch
        
        mystery_file = os.path.basename(mystery_file)
        mystery_file = os.path.splitext(mystery_file)[0]
        
        print(mystery_file)
        
        energy_counts = bin_counts.astype(dtype=np.float64)
        
        # total_counts[file_counter,1] = np.sum(bin_counts[:,1])
        
        # map bin numbers to gamma ray energies
        
        energy_counts[:,0] = mca_coefficients[0] + mca_coefficients[1] * energy_counts[:,0] + mca_coefficients[2] * energy_counts[:,0] ** 2
        smooth_energy_counts = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 6)
        
        # plot smoothed values
        
        plt.plot(energy_counts[:,0], smooth_energy_counts, 'k', linewidth=0.75, alpha=1.0 - 0.6 * (counter / (len(mystery_files) - 1)), label=rf'$T+{epoch:.2f}$' + r' \bf{min}')
        
        counter += 1
        
    plt.legend(prop={'size':14})
    plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
    plt.xticks(fontsize=14)
    plt.xlim(xlim)
    plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
    plt.ylim(ylim)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.gca().set_aspect(aspect)
    plt.savefig(f'figures/mystery_isotope_all_spectra_smooth.png', dpi=400, bbox_inches='tight')
    plt.close()
        
    # load up the data
    
    baseline_bin_counts, baseline_mca_coefficients, _ = loadAsciiSpectrum(mystery_files[0])
    
    baseline_energy_counts = baseline_bin_counts.astype(dtype=np.float64)
    
    # map bin numbers to gamma ray energies
    
    baseline_energy_counts[:,0] = baseline_mca_coefficients[0] + baseline_mca_coefficients[1] * baseline_energy_counts[:,0] + baseline_mca_coefficients[2] * baseline_energy_counts[:,0] ** 2
    baseline_smooth_energy_counts = scipy.ndimage.gaussian_filter1d(baseline_energy_counts[:,1], 6)
    
    # set limits for plotting
        
    xlim = (0,1750)
    ylim = (-50,5)
    aspect = 1.75
    aspect = (xlim[1]-xlim[0]) / (aspect * (ylim[1]-ylim[0]))
    
    counter = 1
    
    plt.plot(list(xlim), [0,0], 'k', linewidth=1.0, label=r'\bf{First Recording}')
        
    for mystery_file in mystery_files[1:]:
        
        # load up the data
    
        bin_counts, mca_coefficients, _ = loadAsciiSpectrum(mystery_file)
        
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
    plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
    plt.xticks(fontsize=14)
    plt.xlim(xlim)
    plt.ylabel(r'$\Delta$ \bf{Detections}', fontsize=16)
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