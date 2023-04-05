from loadAsciiSpectrum import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import glob
from scipy.ndimage import gaussian_filter1d

def backgroundRadiation() -> None:
    
    bin_counts, mca_coefficients = loadAsciiSpectrum(os.path.join('data','calibration_spectrum_bg.Spe'))
    
    energy_counts = bin_counts.astype(dtype=np.float64)
    
    energy_counts[:,0] = mca_coefficients[0] * (energy_counts[:,0] + 1) + mca_coefficients[1] * energy_counts[:,0] ** 2 + mca_coefficients[2] * energy_counts[:,0] ** 3
    
    plt.plot(energy_counts[:,0], energy_counts[:,1])
    plt.show()
    
def mysteryIsotope() -> None:
    
    mystery_files = glob.glob(os.path.join('data','mystery_*.Spe'))
    
    for mystery_file in mystery_files:
        
        bin_counts, mca_coefficients = loadAsciiSpectrum(mystery_file)
    
        energy_counts = bin_counts.astype(dtype=np.float64)
        
        energy_counts[:,0] = mca_coefficients[0] * (energy_counts[:,0] + 1) + mca_coefficients[1] * energy_counts[:,0] ** 2 + mca_coefficients[2] * energy_counts[:,0] ** 3
        
        gaussian_filter1d(energy_counts[:,1], 100)
        
        plt.plot(energy_counts[:,0], energy_counts[:,1])
        
    plt.show()
    
    bg_bin_counts, bg_mca_coefficients = loadAsciiSpectrum(os.path.join('data','calibration_spectrum_bg.Spe'))
    
    bg_energy_counts = bin_counts.astype(dtype=np.float64)
    
    gaussian_filter1d(bg_energy_counts[:,1], 2)
    
    for mystery_file in mystery_files:
        
        bin_counts, mca_coefficients = loadAsciiSpectrum(mystery_file)
    
        energy_counts = bin_counts.astype(dtype=np.float64)
        
        energy_counts[:,1] -= bg_energy_counts[:,1]
        
        energy_counts[:,0] = mca_coefficients[0] * (energy_counts[:,0] + 1) + mca_coefficients[1] * energy_counts[:,0] ** 2 + mca_coefficients[2] * energy_counts[:,0] ** 3
        
        gaussian_filter1d(energy_counts[:,1], 2)
        
        plt.plot(energy_counts[:,0], energy_counts[:,1])
        
    plt.show()

def main() -> None:
    
    backgroundRadiation()
    mysteryIsotope()

if __name__ == '__main__':
    
    main()