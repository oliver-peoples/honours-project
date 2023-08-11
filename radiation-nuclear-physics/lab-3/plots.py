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
        # plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        # plt.gca().set_aspect(aspect)
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
        # plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        # plt.gca().set_aspect(aspect)
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
        # plt.gca().set_aspect(aspect)
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
        # plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        # plt.gca().set_aspect(aspect)
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
        # plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        # plt.gca().set_aspect(aspect)
        plt.savefig(f'figures/{calibration_isotope}_counts_raw_final_mca.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        # plot smoothed values
        
        plt.plot(energy_counts[:,0], smooth_energy_counts, 'k', linewidth=0.75, label=r'$\sigma_{G}=3$ \bf{Spectra}')
        plt.legend(prop={'size':14})
        plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
        plt.xticks(fontsize=14)
        plt.xlim(xlim)
        plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
        # plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        # plt.gca().set_aspect(aspect)
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
        # plt.ylim(ylim)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        # plt.gca().set_aspect(aspect)
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
    # plt.ylim(ylim)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    # plt.gca().set_aspect(aspect)
    plt.savefig('figures/background_counts_raw.png', dpi=400, bbox_inches='tight')
    plt.close()
    
    # plot smoothed values
    
    plt.plot(energy_counts[:,0], smooth_energy_counts, 'k', linewidth=0.75)
    plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
    plt.xticks(fontsize=14)
    plt.xlim(xlim)
    plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
    # plt.ylim(ylim)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    # plt.gca().set_aspect(aspect)
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
    # plt.ylim(ylim)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    # plt.gca().set_aspect(aspect)
    plt.savefig('figures/background_counts_overlay.png', dpi=400, bbox_inches='tight')
    plt.close()
    
def attenuatingMaterial() -> None:
    
    # load up the background data
    
    bg_bin_counts, bg_mca_coefficients, _ = loadAsciiSpectrum(os.path.join('data','calibration_spectrum_bg.Spe'))
    
    bg_energy_counts = bg_bin_counts.astype(dtype=np.float64)
    
    # map bin numbers to gamma ray energies
    
    bg_energy_counts[:,0] = bg_mca_coefficients[0] + bg_mca_coefficients[1] * bg_energy_counts[:,0] + bg_mca_coefficients[2] * bg_energy_counts[:,0] ** 2
    bg_smooth_energy_counts = scipy.ndimage.gaussian_filter1d(bg_energy_counts[:,1], 6)
    bg_smooth_energy_counts *= 2
    
    # attenuator_files = sorted(glob.glob(os.path.join('data','mystery_*.Spe')))
    attenuator_files = sorted(glob.glob(os.path.join('data','attenuator_*.Spe')))
    
    total_counts = np.zeros((len(attenuator_files),2), dtype=np.float64)
    
    file_counter = 0
        
    start_epoch = total_counts[0,0]
    total_counts[:,0] -= total_counts[0,0]
    print(total_counts)
    
    # x_range = total_counts[-1,0] - total_counts[0,0]
    y_max = 10000 * round(np.max(total_counts[:,1]) / 10000) + 5000
    aspect = 1.75
    aspect = (75 / (aspect * y_max))
    
    t_span = np.linspace(0,75,10000)
    theoretical_counts = total_counts[0,1] * np.exp(-9.46e-3 * t_span)
    
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
    
    for attenuator_file in attenuator_files:
        
        # load up the data
    
        bin_counts, mca_coefficients, epoch = loadAsciiSpectrum(attenuator_file)
        
        epoch -= start_epoch
        
        attenuator_file = os.path.basename(attenuator_file)
        attenuator_file = os.path.splitext(attenuator_file)[0]
        
        print(attenuator_file)
        
        energy_counts = bin_counts.astype(dtype=np.float64)
        
        # total_counts[file_counter,1] = np.sum(bin_counts[:,1])
        
        # map bin numbers to gamma ray energies
        
        energy_counts[:,0] = mca_coefficients[0] + mca_coefficients[1] * energy_counts[:,0] + mca_coefficients[2] * energy_counts[:,0] ** 2
        smooth_energy_counts = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 6)
        
        # plot smoothed values
        
        plt.plot(energy_counts[:,0], smooth_energy_counts, 'k', linewidth=0.75, alpha=1.0 - 0.6 * (counter / (len(attenuator_files) - 1)), label=rf'$N={counter}$')
        
        counter += 1
        
    plt.legend(prop={'size':14})
    plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
    plt.xticks(fontsize=14)
    plt.xlim(xlim)
    plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
    # plt.ylim(ylim)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    # plt.gca().set_aspect(aspect)
    plt.savefig(f'figures/attenuator_isotope_all_spectra_smooth.png', dpi=400, bbox_inches='tight')
    plt.close()
    
    # maximum and no attenuation
    
    counter = 0
    
    for attenuator_file in attenuator_files:
        
        if counter == 0 or counter == len(attenuator_files) - 1:
        
            # load up the data
        
            bin_counts, mca_coefficients, epoch = loadAsciiSpectrum(attenuator_file)
            
            epoch -= start_epoch
            
            attenuator_file = os.path.basename(attenuator_file)
            attenuator_file = os.path.splitext(attenuator_file)[0]
            
            print(attenuator_file)
            
            energy_counts = bin_counts.astype(dtype=np.float64)
            
            # total_counts[file_counter,1] = np.sum(bin_counts[:,1])
            
            # map bin numbers to gamma ray energies
            
            energy_counts[:,0] = mca_coefficients[0] + mca_coefficients[1] * energy_counts[:,0] + mca_coefficients[2] * energy_counts[:,0] ** 2
            smooth_energy_counts = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 6)
            
            # plot smoothed values
            
            plt.plot(energy_counts[:,0], smooth_energy_counts, 'k', linewidth=0.75, alpha=1.0 - 0.6 * (counter / (len(attenuator_files) - 1)), label=rf'$N_P={counter}$')
            
        counter += 1
        
    plt.legend(prop={'size':14})
    plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
    plt.xticks(fontsize=14)
    plt.xlim(xlim)
    plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
    # plt.ylim(ylim)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    # plt.gca().set_aspect(aspect)
    plt.savefig(f'figures/attenuated_unattenuated_spectra_smooth.png', dpi=400, bbox_inches='tight')
    plt.close()
        
    # load up the data
    
    baseline_bin_counts, baseline_mca_coefficients, _ = loadAsciiSpectrum(attenuator_files[0])
    
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
    
    plt.plot(list(xlim), [0,0], 'k', linewidth=1.0, label=r'$N=0$')
        
    for attenuator_file in attenuator_files[1:]:
        
        # load up the data
    
        bin_counts, mca_coefficients, _ = loadAsciiSpectrum(attenuator_file)
        
        attenuator_file = os.path.basename(attenuator_file)
        attenuator_file = os.path.splitext(attenuator_file)[0]
        
        energy_counts = bin_counts.astype(dtype=np.float64)
        
        # map bin numbers to gamma ray energies
        
        energy_counts[:,0] = mca_coefficients[0] + mca_coefficients[1] * energy_counts[:,0] + mca_coefficients[2] * energy_counts[:,0] ** 2
        smooth_energy_counts = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 6)
        
        # plot smoothed values
        
        plt.plot(energy_counts[:,0], smooth_energy_counts - baseline_smooth_energy_counts, 'k', linewidth=0.75, alpha=0.9 - 0.4 * (counter / (len(attenuator_files) - 1)))
        
        counter += 1
        
    plt.legend(prop={'size':14})
    plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
    plt.xticks(fontsize=14)
    plt.xlim(xlim)
    plt.ylabel(r'$\Delta$ \bf{Detections}', fontsize=16)
    # plt.ylim(ylim)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    # plt.gca().set_aspect(aspect)
    plt.savefig(f'figures/attenuator_difference_counts_smooth.png', dpi=400, bbox_inches='tight')
    plt.close()
    
def aluminumAttenuator() -> None:
    
    # load up the background data
    
    bg_bin_counts, bg_mca_coefficients, _ = loadAsciiSpectrum(os.path.join('data','calibration_spectrum_bg.Spe'))
    
    bg_energy_counts = bg_bin_counts.astype(dtype=np.float64)
    
    # map bin numbers to gamma ray energies
    
    x_axis = bg_energy_counts[:,0]
    x_axis = bg_mca_coefficients[0] + bg_mca_coefficients[1] * x_axis + bg_mca_coefficients[2] * x_axis ** 2
    
    # attenuator_files = sorted(glob.glob(os.path.join('data','mystery_*.Spe')))
    al_p_files = sorted(glob.glob(os.path.join('data','Al_*_p.Spe')))
    al_np_files = sorted(glob.glob(os.path.join('data','Al_*_np.Spe')))
    
    al_p_data = []
    al_np_data = []
    
    for al_p_file in al_p_files:
        
        bin_counts, mca_coefficients, _ = loadAsciiSpectrum(al_p_file)
        
        al_p_file = os.path.basename(al_p_file)
        al_p_file = os.path.splitext(al_p_file)[0]
        
        energy_counts = bin_counts.astype(dtype=np.float64)
        
        # map bin numbers to gamma ray energies
        
        energy_counts[:,0] = mca_coefficients[0] + mca_coefficients[1] * energy_counts[:,0] + mca_coefficients[2] * energy_counts[:,0] ** 2
        smooth_energy_counts = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 6)
        
        al_p_data.append(smooth_energy_counts)
        
    for al_np_file in al_np_files:
        
        bin_counts, mca_coefficients, _ = loadAsciiSpectrum(al_np_file)
        
        al_np_file = os.path.basename(al_np_file)
        al_np_file = os.path.splitext(al_np_file)[0]
        
        energy_counts = bin_counts.astype(dtype=np.float64)
        
        # map bin numbers to gamma ray energies
        
        energy_counts[:,0] = mca_coefficients[0] + mca_coefficients[1] * energy_counts[:,0] + mca_coefficients[2] * energy_counts[:,0] ** 2
        smooth_energy_counts = scipy.ndimage.gaussian_filter1d(energy_counts[:,1], 6)
        
        al_np_data.append(smooth_energy_counts)
        
    avg_p_data = np.mean(np.array([*al_p_data]), axis=0)
    avg_np_data = np.mean(np.array([*al_np_data]), axis=0)
    
    plt.plot(x_axis, avg_p_data, label=r'\bf{Attenuated}')
    plt.plot(x_axis, avg_np_data, label=r'\bf{Unattenuated}')
    
    xlim = (0,1750)
    
    plt.legend(prop={'size':14})
    plt.xlabel(r'\bf{Gamma Ray Energy (keV)}',fontsize=16)
    plt.xticks(fontsize=14)
    plt.xlim(xlim)
    plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
    # plt.ylim(ylim)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    # plt.gca().set_aspect(aspect)
    plt.savefig(f'figures/aluminum_counts_smooth.png', dpi=400, bbox_inches='tight')
    plt.close()
        
def main() -> None:
    
    # calibrationIsotopes()
    # backgroundRadiation()
    # attenuatingMaterial()
    # aluminumAttenuator()
    
    counts = np.genfromtxt('counts.csv', delimiter=',')
    thicknesses = np.genfromtxt('thicknesses.csv', delimiter=',')
    
    thicknesses = [np.sum(thicknesses[0:idx,1]) for idx in range(1,np.shape(thicknesses)[0] + 1)]
    
    plt.plot(thicknesses, counts[:,1])
    plt.yscale('log')
    plt.xlabel(r'\bf{Attenuator Thickness (mm)}',fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'figures/cesium_peak_area_lead.png', dpi=400, bbox_inches='tight')
    plt.close()
    
    for recording_idx in range(np.shape(thicknesses)[0]):
        print(f'({thicknesses[recording_idx]},{np.log10(counts[recording_idx,1]) - np.log10(counts[0,1])})')
    
    plt.plot(thicknesses, counts[:,2])
    plt.yscale('log')
    plt.xlabel(r'\bf{Attenuator Thickness (mm)}',fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel(r'\bf{\# of Detections}', fontsize=16)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'figures/cobalt_peak_area_lead.png', dpi=400, bbox_inches='tight')
    plt.close()
    
    for recording_idx in range(np.shape(thicknesses)[0]):
        print(f'({thicknesses[recording_idx]},{np.log10(counts[recording_idx,2]) - np.log10(counts[0,2])})')

if __name__ == '__main__':
    
    main()