# from scipy.special import hermite as physicistsHermite
import numpy as np
# from parula import parula
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import RegularPolygon, Circle
import os

# from scipy.spatial import ConvexHull, convex_hull_plot_2d

cm = 1/2.54

path = os.path.dirname(__file__)

matplotlib.rcParams['text.usetex'] = True

CONCENTRIC = 1
STAGGERED = 2

def main() -> None:
    
    method = int(input('concentric (1) or staggered (2)? '))
    
    if method == STAGGERED:
        
        X_DIFF = float(input('X_DIFF: '))
        
        ROWS = int(input('ROWS: '))
        COLS = int(input('COLS: '))
        
        G2_CAPABLE_IDX = []
        
        try:
            G2_CAPABLE_IDX = [int(idx) for idx in input('G2_CAPABLE_IDX: ').split(',')]
        except:
            G2_CAPABLE_IDX = [-1]
        
        WAISTS = float(input('WAISTS: '))
        
        DRAW_HEXAGONS = input('Draw cores? (y/n): ')
    
        ROW_SHIFT = X_DIFF / 2
        
        print(ROW_SHIFT)

        R_VERTEX = ROW_SHIFT * 2 / 3**0.5

        Y_DIFF = 1.5 * R_VERTEX

        col_indexes = np.arange(COLS)
        row_indexes = np.arange(ROWS)

        cores = np.ndarray((ROWS * COLS,3), dtype=np.float64)

        cores[:,2] = 0

        for row in range(ROWS):
            
            for col in range(COLS):
                
                cores[row * COLS:(row + 1) * COLS,0] = range(COLS)
                cores[row * COLS:(row + 1) * COLS,1] = row

        cores[:,0] *= X_DIFF
        cores[:,0] += np.mod(cores[:,1], 2) * ROW_SHIFT
        cores[:,0] -= (ROW_SHIFT + X_DIFF * (COLS - 1)) / 2
        cores[:,1] *= Y_DIFF
        cores[:,1] -= (Y_DIFF * (ROWS - 1)) / 2
            
        G1_ONLY_IDX = [idx for idx in range(np.shape(cores)[0]) if idx not in G2_CAPABLE_IDX]
            
            # g2_capable = np.genfromtxt(os.path.join(path,'g2_capable_indexes.csv'), delimiter=',', dtype=np.int8, skip_header=1)
        
        if G2_CAPABLE_IDX[0] != -1:
            plt.scatter(cores[G2_CAPABLE_IDX,0], cores[G2_CAPABLE_IDX,1], c='black', marker='+', linewidths=0.5, s=10)
        
        if len(G1_ONLY_IDX) > 0:
            # g1_only = [ idx for idx in range(np.shape(cores)[0]) if idx not in G2_CAPABLE_IDX[:,1]]
            plt.scatter(cores[G1_ONLY_IDX,0], cores[G1_ONLY_IDX,1], c='black', marker='.', s=10)
            
        if DRAW_HEXAGONS == 'y':
            
            circles = input ('Draw cores as circles? (y/n): ')
            
            for core in cores:
                # fix radius here
                
                shape = 'patch_type'
                
                if circles == 'n':
                    
                    shape = RegularPolygon((core[0], core[1]), numVertices=6, radius=X_DIFF * 0.5, alpha=0.2, edgecolor='k')
                
                else: 
                    
                    shape = Circle((core[0],core[1]), radius=X_DIFF * 0.45, alpha=0.2, edgecolor='k')
                
                plt.gca().add_patch(shape)

        plt.xlabel(r'$x/\sigma$', fontsize=10)
        plt.xlim(-WAISTS,WAISTS)
        plt.xticks(fontsize=10)
        plt.ylabel(r'$y/\sigma$', fontsize=10)
        plt.ylim(-WAISTS,WAISTS)
        plt.xticks(fontsize=10)
        plt.gca().set_aspect(1)
        plt.tight_layout()
        plt.gcf().set_figwidth(val=0.32 * 15.3978 * cm)
        plt.savefig(os.path.join(path,f'staggered_r_{ROWS}_c_{COLS}_g2_' + ''.join([(str(g2_idx) + '_') for g2_idx in G2_CAPABLE_IDX]) + 'plot.png'), dpi=500, bbox_inches='tight')
        plt.close()
        
    elif method == CONCENTRIC:
        
        # H(n) = n^3 - (n-1)^3 = 3n(n-1)+1 = 3n^2 - 3n +1.
        
        concentric_rings = int(input('concentric rings: '))
        intercore_distance = float(input('intercore distance: '))
        G2_CAPABLE_IDX = []
        
        try:
            G2_CAPABLE_IDX = [int(idx) for idx in input('G2_CAPABLE_IDX: ').split(',')]
        except:
            G2_CAPABLE_IDX = [-1]
            
        WAISTS = float(input('WAISTS: '))

        DRAW_HEXAGONS = input('Draw cores? (y/n): ')
        
        n = concentric_rings + 1

        num_cores = 3 * (n * n) - 3 * n + 1

        cores = np.ndarray((num_cores, 3), dtype=np.float64)

        cores[0,:] = np.array([0,0,0])

        idx_accumulator = 1

        for ring_num in range(1,n):
            
            ring_points = int(
                (3 * ((ring_num + 1) * (ring_num + 1)) - 3 * (ring_num + 1) + 1)
                -
                (3 * (ring_num * ring_num) - 3 * ring_num + 1)
            )

            bridge_points = int((ring_points - 6) / 6)

            bl_vertex = np.array([-0.5,-0.5 * 3**0.5]).T
        
            bl_vertex *= float(ring_num)

            for vertex_point_num in range(6):
            
                rotation_mat = np.array([
                    [np.cos(vertex_point_num * np.pi / 3),-np.sin(vertex_point_num * np.pi / 3)],
                    [np.sin(vertex_point_num * np.pi / 3),np.cos(vertex_point_num * np.pi / 3)]
                ], dtype=np.float64)

                vertex_point = rotation_mat @ bl_vertex

                cores[idx_accumulator,:] = np.array([vertex_point[0],vertex_point[1],0])
                cores[idx_accumulator,:] *= intercore_distance

                idx_accumulator += 1

                for non_vertex_point in range(bridge_points):
                
                    base_position = bl_vertex.copy()

                    base_position[0] += ring_num * (non_vertex_point + 1) * 1. / (bridge_points + 1)

                    side_point = rotation_mat @ base_position

                    cores[idx_accumulator,:] = np.array([side_point[0],side_point[1],0.])
                    cores[idx_accumulator,:] *= intercore_distance

                    idx_accumulator += 1
                    
        G1_ONLY_IDX = [idx for idx in range(np.shape(cores)[0]) if idx not in G2_CAPABLE_IDX]
            
            # g2_capable = np.genfromtxt(os.path.join(path,'g2_capable_indexes.csv'), delimiter=',', dtype=np.int8, skip_header=1)
        
        if G2_CAPABLE_IDX[0] != -1:
            plt.scatter(cores[G2_CAPABLE_IDX,0], cores[G2_CAPABLE_IDX,1], c='black', marker='+', linewidths=0.5, s=10)
        
        if len(G1_ONLY_IDX) > 0:
            # g1_only = [ idx for idx in range(np.shape(cores)[0]) if idx not in G2_CAPABLE_IDX[:,1]]
            plt.scatter(cores[G1_ONLY_IDX,0], cores[G1_ONLY_IDX,1], c='black', marker='.', s=10)
            
        if DRAW_HEXAGONS == 'y':
            
            circles = input ('Draw cores as circles? (y/n): ')
            
            for core in cores:
                # fix radius here
                
                shape = 'patch_type'
                
                if circles == 'n':
                    
                    shape = RegularPolygon((core[0], core[1]), numVertices=6, radius=intercore_distance * 0.5, alpha=0.2, edgecolor='k')
                
                else: 
                    
                    shape = Circle((core[0],core[1]), radius=intercore_distance * 0.45, alpha=0.2, edgecolor='k')
                
                plt.gca().add_patch(shape)

        plt.xlabel(r'$x/\sigma$', fontsize=10)
        plt.xlim(-WAISTS,WAISTS)
        plt.xticks(fontsize=10)
        plt.ylabel(r'$y/\sigma$', fontsize=10)
        plt.ylim(-WAISTS,WAISTS)
        plt.xticks(fontsize=10)
        plt.gca().set_aspect(1)
        plt.tight_layout()
        plt.gcf().set_figwidth(val=0.32 * 15.3978 * cm)
        plt.savefig(os.path.join(path,f'concentric_{concentric_rings}_g2_' + ''.join([(str(g2_idx) + '_') for g2_idx in G2_CAPABLE_IDX]) + 'plot.png'), dpi=500, bbox_inches='tight')
        plt.close()
                
    
    
if __name__ == '__main__':
    
    main()