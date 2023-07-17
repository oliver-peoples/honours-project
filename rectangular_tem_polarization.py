import qclm
import numpy as np
import pathos.multiprocessing as pmp
from scipy.special import hermite as physicistsHermite
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib
from parula import parula

matplotlib.rcParams['text.usetex'] = True

sqrt_2 = np.sqrt(2)

def main():
    
    x_size = 750
    y_size = 750
    
    m = 0
    n = 1
    
    hermite_m = physicistsHermite(m)
    hermite_n = physicistsHermite(n)
    
    x_linspace = np.linspace(-1.0, 1.0, x_size)
    y_linspace = np.linspace(-1.0, 1.0, y_size)
    
    x_meshgrid, y_meshgrid = np.meshgrid(x_linspace, y_linspace)
    
    x_meshgrid = x_meshgrid.flatten()
    y_meshgrid = y_meshgrid.flatten()
    
    def cmplxE_mn(idx):
    
        x = x_meshgrid[idx]
        y = y_meshgrid[idx]
        
        hermite_component = hermite_m(sqrt_2 * x) * hermite_n(sqrt_2 * y) 
        
        cmplx_exp = np.exp(-(x**2 + y**2)*1j - 1j - 1j * (m + n + 1) * np.pi)
        
        return np.angle(hermite_component * cmplx_exp)
    
    mp_pool = pmp.Pool(processes=pmp.cpu_count())
    phases = mp_pool.map(cmplxE_mn, range(x_size * y_size))
    mp_pool.close()
    
    phase_matrix = np.zeros(shape=(y_size,x_size))
    
    for row in range(y_size):
        
        for col in range(x_size):
            
            idx_1d = row * x_size + col
            
            phase_matrix[row,col] = phases[idx_1d]
            
    heatmap = plt.pcolormesh(x_linspace, y_linspace, phase_matrix, cmap=parula)
    plt.xlabel(r"$x/w$", fontsize=18)
    plt.ylabel(r"$y/w$", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    cbar = plt.colorbar(heatmap, pad=0.01, ticks=[-np.pi/2,0,np.pi/2])
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(r'$\theta$', fontsize=18, rotation=0, labelpad=15)
    
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(f'tem_m_{m}_n_{n}_phases.png', dpi=600, bbox_inches='tight')
    plt.close()
    
    arg_vec = [(m, n, y_val, x_size, (-1.0,1.0)) for y_val in y_linspace]
            
    mp_pool = pmp.Pool(processes=pmp.cpu_count())
    
    intensity_vals = np.stack(mp_pool.starmap(qclm.parallelTEM, arg_vec), axis=0)
            
    plt.title(r'$\mathrm{m}=' + str(m) + r',\mathrm{n}=' + str(n) + r'$', fontsize=28)
    plt.pcolormesh(x_linspace, y_linspace, intensity_vals / np.max(intensity_vals), cmap=parula)
    plt.xlabel(r"$x/w$", fontsize=24)
    plt.ylabel(r"$y/w$", fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    cbar = plt.colorbar(pad=0.01)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(r"$I\left(x,y\right)/I_{max}$", fontsize=24, rotation=-90, labelpad=28)
    # cbar.set_ticks
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(f'tem_m_{m}_n_{n}_norm_intensities.png', dpi=600, bbox_inches='tight')
    plt.close()
    
if __name__ == '__main__':
    
    main()