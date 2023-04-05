# Oliver Kirkpatrick - oliver-peoples - 10.133.8.27 - finn@finna - Razer Blade Stealth 15
# CONDA_ENV - Python 3.10.9 - Balloon Artificial Research Environment for Navigation and Autonomy (BallARENA) TASDCRC
# 5/4/23 16:56

from typing import Tuple, List
import numpy as np

def lineFlag(file, flag_word):
    cur_pos = file.tell()
    flag_word_present = file.readline().strip('\n') == flag_word
    file.seek(cur_pos)
    return flag_word_present

def returnBinIndices(line: str) -> Tuple:
    
    line = line.strip('\n')
    
    bin_indices = tuple([int(bin_index) for bin_index in line.split(' ')])
    
    return bin_indices, bin_indices[1] - bin_indices[0] + 1

def loadAsciiSpectrum(path: str):
    
    print(f'Loading ascii Spe file at {path}')
    
    f = open(path, 'r')
    
    line: str
    
    while not lineFlag(f, '$DATA:'):
        
        line = f.readline()
        
    f.readline()
    
    bin_indices, num_bins = returnBinIndices(f.readline())
    
    bin_counts = np.zeros((num_bins, 2), dtype=np.int16)
    
    bin_counts[:,0] = range(bin_indices[0], bin_indices[1] + 1)
    
    bin_num = 0
    
    while bin_num < num_bins:
        
        bin_counts[bin_num,1] = int(f.readline().strip('\n'))
        
        bin_num += 1
        
    while not lineFlag(f, '$MCA_CAL:'):
        
        print(f.readline())
        
    f.readline()
    f.readline()
    
    mca_coefficients = [float(coefficient) for coefficient in f.readline().strip().split(' ')[:-1]]     
    
    return bin_counts, mca_coefficients   