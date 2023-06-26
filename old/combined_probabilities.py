import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['text.usetex'] = True

def main() -> None:
    
    w = 1.
    
    emission_psf = np.genfromtxt('tem_m_0_n_0.csv', delimiter=',')
    
    y_linspace = np.linspace(-2.0, 2.0, np.shape(emission_psf)[0])
    x_linspace = np.linspace(-2.0, 2.0, np.shape(emission_psf)[1])
    
    for m in range(0,4):
        
        for n in range(0,4):
            
            excitation_psf = np.genfromtxt(f'tem_m_{m}_n_{n}.csv', delimiter=',')
            
            total_psf = np.multiply(excitation_psf, emission_psf)
            
            total_psf *= 1. / np.max(total_psf)
                    
            exec('plt.title(r"$\mathbf{TEM}_{0,0}\circ\mathbf{TEM}_{' + str(m) + ',' + str(n) + '}$", fontsize=16)')
            plt.pcolormesh(x_linspace / w, y_linspace / w, total_psf, cmap='Blues')
            plt.xlabel(r"$x/w$", fontsize=16)
            plt.ylabel(r"$y/w$", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            cbar = plt.colorbar()
            cbar.set_label(r"$P_{x,y}/P_{max}$", fontsize=16, rotation=-90, labelpad=20)
            # cbar.set_ticks
            plt.gca().set_aspect(1)
            plt.tight_layout()
            plt.savefig(f'total_psf_m_{m}_n_{n}.png', dpi=1200, bbox_inches='tight')
            plt.close()
            
    
if __name__ == '__main__':
    
    main()