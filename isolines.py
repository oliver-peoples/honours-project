
import numpy as np
from parula import parula
import matplotlib.pyplot as plt
import matplotlib
from qclm import Emitter, GaussLaguerre, GaussHermite, Detector

matplotlib.rcParams['text.usetex'] = True


def main() -> None:
    
    # our detector 
    
    detector = Detector()
    
    # light structures
    
    illumination_structures = [
        GaussHermite(0, 0, 1., 0.5),
        GaussHermite(0, 1, 1., 0.5, rotation=0),
        GaussHermite(0, 1, 1., 0.5, rotation=np.pi/6),
        GaussHermite(0, 1, 1., 0.5, rotation=np.pi/3),
        GaussHermite(0, 1, 1., 0.5, rotation=np.pi/2)
    ]
    
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

    grid_x = 3750
    grid_y = 3750
    waists = 3
    
    x_linspace = np.linspace(-detector.waist/2,detector.waist/2, grid_x)
    y_linspace = np.linspace(-detector.waist/2,detector.waist/2, grid_y)
    
    x_meshgrid, y_meshgrid = np.meshgrid(
        x_linspace,
        y_linspace
    )
    
    total_error = np.zeros_like(x_meshgrid)
    
    for illumination_structure in illumination_structures:
        
        print(f'{illumination_structure.modeTypeString().lower()}_{illumination_structure.orderString()}')
        
        # intensity field
        
        intensity_map = illumination_structure.intensityMap(x_meshgrid, y_meshgrid)
        
        # volume = np.trapz(
        #     y=np.asarray(
        #         [np.trapz(y=intensity_row, x=x_linspace) for intensity_row in intensity_map[:]]
        #     ),
        #     x=y_linspace
        # )
        
        # print(volume)
        # volume = 1.
        # intensity_map /= np.abs(volume)

        plt.title(r'$\mathrm{' + illumination_structure.modeTypeString() + r'}_{' + f'{illumination_structure.orderString()}' + r'}$', fontsize=24, pad=10)
        plt.pcolormesh(x_meshgrid, y_meshgrid, intensity_map, cmap=parula)
        plt.xlabel(r"$x$", fontsize=18)
        plt.ylabel(r"$y$", fontsize=18)
        plt.xticks(fontsize=4 + 12)
        plt.yticks(fontsize=4 + 12)
        cbar = plt.colorbar(pad=0.01)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(r"$I_{pl}\left(\rho,\phi\right)/I_{max}$", fontsize=18, rotation=-90, labelpad=38)
        plt.scatter(e_1.xyz[0], e_1.xyz[1], c='r', marker='x', s=40, linewidths=1)
        plt.scatter(e_2.xyz[0], e_2.xyz[1], c='r', marker='x', s=40, linewidths=1)
        plt.gca().set_aspect(1)
        plt.tight_layout()
        plt.savefig(f'isolines/{illumination_structure.modeTypeString().lower()}_{illumination_structure.orderString()}_i_xy.png', dpi=400, bbox_inches='tight')
        plt.close()
    
        # true detector intensities

        p_1_true = e_1.relative_brightness * (illumination_structure.intensityFn(*e_1.xyz))
        p_2_true = e_2.relative_brightness * (illumination_structure.intensityFn(*e_2.xyz))
        
        p_1_true = detector.detectFn(e_1.xyz[0:2], p_1_true)
        p_2_true = detector.detectFn(e_2.xyz[0:2], p_2_true)
        
        g_1_true = (p_1_true + p_2_true) / (e_1.relative_brightness + e_2.relative_brightness)
        
        alpha = p_2_true / p_1_true
        
        g_2_true = (2 * alpha) / (1 + alpha)**2
        
        # guessed values
        
        p_2_guessed = e_2.relative_brightness * (illumination_structure.intensityFn(x_meshgrid, y_meshgrid))
        p_2_guessed = detector.detectFn(x_meshgrid, y_meshgrid, p_2_guessed)
        
        g_1_guess = (p_1_true + p_2_guessed) / (e_1.relative_brightness + e_2.relative_brightness)
        
        plt.title(r'$\mathrm{' + illumination_structure.modeTypeString() + r'}_{' + f'{illumination_structure.orderString()}' + r'}$', fontsize=24, pad=10)
        plt.pcolormesh(x_meshgrid, y_meshgrid, g_1_guess, cmap=parula)
        plt.xlabel(r"$x$", fontsize=18)
        plt.ylabel(r"$y$", fontsize=18)
        plt.xticks(fontsize=4 + 12)
        plt.yticks(fontsize=4 + 12)
        cbar = plt.colorbar(pad=0.01)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(r"$\mathsf{G}_{2}^{(1)}$", fontsize=18, rotation=-90, labelpad=38)
        plt.scatter(e_1.xyz[0], e_1.xyz[1], c='r', marker='x', s=40, linewidths=1)
        plt.scatter(e_2.xyz[0], e_2.xyz[1], c='r', marker='x', s=40, linewidths=1)
        plt.gca().set_aspect(1)
        plt.tight_layout()
        plt.savefig(f'isolines/{illumination_structure.modeTypeString().lower()}_{illumination_structure.orderString()}_g_1.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        alpha = p_2_guessed / p_1_true
        
        g_2_guess = (2 * alpha) / (1 + alpha)**2 
        
        plt.title(r'$\mathrm{' + illumination_structure.modeTypeString() + r'}_{' + f'{illumination_structure.orderString()}' + r'}$', fontsize=24, pad=10)
        plt.pcolormesh(x_meshgrid, y_meshgrid, g_2_guess, cmap=parula)
        plt.xlabel(r"$x$", fontsize=18)
        plt.ylabel(r"$y$", fontsize=18)
        plt.xticks(fontsize=4 + 12)
        plt.yticks(fontsize=4 + 12)
        cbar = plt.colorbar(pad=0.01)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(r"$\mathsf{G}_{2}^{(2)}$", fontsize=18, rotation=-90, labelpad=38)
        plt.scatter(e_1.xyz[0], e_1.xyz[1], c='r', marker='x', s=40, linewidths=1)
        plt.scatter(e_2.xyz[0], e_2.xyz[1], c='r', marker='x', s=40, linewidths=1)
        plt.gca().set_aspect(1)
        plt.tight_layout()
        plt.savefig(f'isolines/{illumination_structure.modeTypeString().lower()}_{illumination_structure.orderString()}_g_2.png', dpi=400, bbox_inches='tight')
        plt.close()
        
        g_1_guess -= g_1_true
        g_2_guess -= g_2_true
        
        sum_squares = g_1_guess**2 + g_2_guess**2
        sum_squares -= np.min(sum_squares)
        
        total_error += sum_squares
        
        plt.title(r'$\mathrm{' + illumination_structure.modeTypeString() + r'}_{' + f'{illumination_structure.orderString()}' + r'}$', fontsize=24, pad=10)
        plt.pcolormesh(x_meshgrid, y_meshgrid, sum_squares, cmap=parula)
        plt.xlabel(r"$x$", fontsize=18)
        plt.ylabel(r"$y$", fontsize=18)
        plt.xticks(fontsize=4 + 12)
        plt.yticks(fontsize=4 + 12)
        plt.gca().contour(x_meshgrid, y_meshgrid, sum_squares, linewidths=1, colors='k', levels=[np.min(sum_squares) + 0.000005])
        cbar = plt.colorbar(pad=0.01)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(r"$\left(\mathcal{G}_{2}^{(1)}-\mathsf{G}_{2}^{(1)}\right)^{2}+\left(\mathcal{G}_{2}^{(2)}-\mathsf{G}_{2}^{(2)}\right)^{2}$", fontsize=18, rotation=-90, labelpad=38)
        plt.scatter(e_1.xyz[0], e_1.xyz[1], c='r', marker='x', s=40, linewidths=1)
        plt.scatter(e_2.xyz[0], e_2.xyz[1], c='r', marker='x', s=40, linewidths=1)
        plt.gca().set_aspect(1)
        plt.tight_layout()
        plt.savefig(f'isolines/{illumination_structure.modeTypeString().lower()}_{illumination_structure.orderString()}_sum_squares.png', dpi=400, bbox_inches='tight')
        plt.close()
        
    print(np.min(total_error))
        
    # plt.title(r'$\mathrm{' + illumination_structure.modeTypeString() + r'}_{' + f'{illumination_structure.orderString()}' + r'}$', fontsize=24, pad=10)
    plt.pcolormesh(x_meshgrid, y_meshgrid, total_error, cmap=parula)
    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel(r"$y$", fontsize=18)
    plt.xticks(fontsize=4 + 12)
    plt.yticks(fontsize=4 + 12)
    plt.gca().contour(x_meshgrid, y_meshgrid, total_error, linewidths=1, colors='k', levels=[np.min(total_error) + 0.000005])
    cbar = plt.colorbar(pad=0.01)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(r"$\mathrm{Sum\;Square\;Error}$", fontsize=18, rotation=-90, labelpad=38)
    plt.scatter(e_1.xyz[0], e_1.xyz[1], c='r', marker='x', s=40, linewidths=1)
    plt.scatter(e_2.xyz[0], e_2.xyz[1], c='r', marker='x', s=40, linewidths=1)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(f'isolines/total_error.png', dpi=400, bbox_inches='tight')
    plt.close()
        
        
    
if __name__ == '__main__':
    
    main()