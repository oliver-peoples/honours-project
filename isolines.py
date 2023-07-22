from scipy.special import genlaguerre as genLaguerre
import numpy as np
from dataclasses import dataclass
from parula import parula
import matplotlib.pyplot as plt
import matplotlib
from qclm import Emitter

matplotlib.rcParams['text.usetex'] = True

@dataclass
class GaussLaguerre:
    
    p: int
    l: int
    
    I_0: float
    waist: float
    
    center: np.ndarray = np.array([0.,0.])
    
    def __post_init__(self) -> None:
        
        self.laguerre = genLaguerre(self.p, self.l)
        
    def rhoFn(self, r) -> float:
    
        return 2. * r**2 / self.waist**2
        
    def intensityFn(self, x: float, y: float):
        
        relative_x = x - self.center[0]
        relative_y = y - self.center[1]
        
        r = np.sqrt(relative_x**2 + relative_y**2)
        phi = np.arctan2(relative_y, relative_x)
        
        rho = self.rhoFn(r)
        
        gl_tem = self.I_0 * (rho**self.l) * (self.laguerre(rho)**2) * (np.cos(self.l * phi)**2) * np.exp(-rho)
        
        return gl_tem
    
@dataclass
class Detector:
    
    waist = 1.
    
    center: np.ndarray = np.array([0.,0.])
    
    def detectFn(self, x: float, y: float, p: float=1.):
        
        relative_x = x - self.center[0]
        relative_y = y - self.center[1]
        
        r = np.sqrt(relative_x**2 + relative_y**2)
        
        return np.exp(-(r**2 / 2) / (2 * self.waist**2)) * p

def main() -> None:
    
    # our detector 
    
    detector = Detector()
    
    # light structures
    
    gl = GaussLaguerre(1, 3, 1., 1.)
    
    # emitters
    
    e_1 = Emitter(
        np.array([-0.3,0.2]),
        1.0
    )

    e_2 = Emitter(
        np.array([0.2,0.1]),
        0.5
    )
    
    # plotting stuff

    grid_x = 1750
    grid_y = 1750
    waists = 3
    
    x_meshgrid, y_meshgrid = np.meshgrid(
        np.linspace(-waists * gl.waist, waists * gl.waist, grid_x),
        np.linspace(waists * gl.waist, -waists * gl.waist, grid_y)
    )
    
    # true detector intensities

    p_1_true = e_1.relative_brightness * gl.intensityFn(*e_1.xy)
    p_2_true = e_2.relative_brightness * gl.intensityFn(*e_2.xy)
    
    p_1_true = detector.detectFn(*e_1.xy, p_1_true)
    p_2_true = detector.detectFn(*e_2.xy, p_2_true)
    
    g_1_true = (p_1_true + p_2_true) / (e_1.relative_brightness + e_1.relative_brightness)
    
    alpha = p_2_true / p_1_true
    
    g_2_true = (2 * alpha) / (1 + alpha)**2
    
    print(g_1_true, g_2_true)
    
    # guessed values
    
    

    # plt.title(r'$\mathrm{GL}_{pl},\;p=' + str(gl.p) + r',\;l=' + str(gl.l) + r'$', fontsize=28, pad=10)
    # plt.pcolormesh(x_meshgrid / waist, y_meshgrid / waist, gl_tem, cmap=parula)
    # plt.xlabel(r"$x/w$", fontsize=24)
    # plt.ylabel(r"$y/w$", fontsize=24)
    # plt.xticks(fontsize=22)
    # plt.yticks(fontsize=22)
    # cbar = plt.colorbar(pad=0.01)
    # cbar.ax.tick_params(labelsize=18)
    # cbar.set_label(r"$I_{pl}\left(\rho,\phi\right)/I_{max}$", fontsize=24, rotation=-90, labelpad=28)
    # plt.gca().set_aspect(1)
    # plt.tight_layout()
    # plt.savefig(f'gl-modes/test_p_{gl.p}_l_{gl.l}_i.png', dpi=400, bbox_inches='tight')
    # plt.close()
    
if __name__ == '__main__':
    
    main()