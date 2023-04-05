from loadAsciiSpectrum import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

def backgroundRadiation() -> None:
    
    bin_counts, mca_coefficients = loadAsciiSpectrum(os.path.join('data','calibration_spectrum_bg.Spe'))
    
    energy_counts = bin_counts.astype(dtype=np.float64)
    
    energy_counts[:,0] = mca_coefficients[0] + mca_coefficients[1] * energy_counts[:,0] + mca_coefficients[2] * energy_counts[:,0] ** 2
    
    plt.plot(energy_counts[:,0], energy_counts[:,1])
    plt.show()
    
# def mysteryIsotope() -> None:
    
#     for mystery_idx in range(1,6)

def main() -> None:
    
    backgroundRadiation()

if __name__ == '__main__':
    
    main()