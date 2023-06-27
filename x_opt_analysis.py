import qclm
import numpy as np
import pathos.multiprocessing as pmp
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
from parula import parula

matplotlib.rcParams['text.usetex'] = True

x_opt = np.genfromtxt('x_opt_position.csv', delimiter=',')

emitter_separation = np.linalg.norm(x_opt[0,:2] - x_opt[0,2:])

plot_center = np.array([0.5 * (x_opt[0,:2][0] + x_opt[0,2:][0]), 0.5 * (x_opt[0,:2][1] + x_opt[0,2:][1])])

x_range = (-0.8 * emitter_separation + plot_center[0],0.8 * emitter_separation + plot_center[0])
y_range = (-0.8 * emitter_separation + plot_center[1],0.8 * emitter_separation + plot_center[1])

plt.scatter(x_opt[2:,0],x_opt[2:,1], c='b', s=2., marker='.')
plt.scatter(x_opt[2:,2],x_opt[2:,3], c='r', s=2., marker='.')
plt.scatter(x_opt[0,0],x_opt[0,1], c='k', marker='x', s=40, linewidths=1.0)
plt.scatter(x_opt[0,2],x_opt[0,3], c='k', marker='x', s=40, linewidths=1.0)
plt.scatter(x_opt[1,0],x_opt[1,1], c='k', s=40, marker='+', linewidths=0.5)
plt.scatter(x_opt[1,2],x_opt[1,3], c='k', s=40, marker='+', linewidths=0.5)    
plt.xlim(*x_range)
plt.ylim(*y_range)
plt.xlabel(r"$x/w$", fontsize=18)
plt.ylabel(r"$y/w$", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().set_aspect(1)
plt.tight_layout()
plt.savefig('x_opt_analysis.png', dpi=600, bbox_inches='tight')
plt.close()

plt.scatter(x_opt[2:,0],x_opt[2:,1], c='b', s=2., marker='.')
plt.scatter(x_opt[2:,2],x_opt[2:,3], c='r', s=2., marker='.')
plt.scatter(x_opt[0,0],x_opt[0,1], c='k', marker='x', s=40, linewidths=1.0)
plt.scatter(x_opt[0,2],x_opt[0,3], c='k', marker='x', s=40, linewidths=1.0)
plt.scatter(x_opt[1,0],x_opt[1,1], c='k', s=40, marker='+', linewidths=0.5)
plt.scatter(x_opt[1,2],x_opt[1,3], c='k', s=40, marker='+', linewidths=0.5)    
plt.xlim((plot_center[0],x_range[1]))
plt.ylim((-0.3 * emitter_separation + plot_center[1],plot_center[1]))
plt.xlabel(r"$x/w$", fontsize=18)
plt.ylabel(r"$y/w$", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().set_aspect(1)
plt.tight_layout()
plt.savefig('x_opt_quadrant_raw.png', dpi=600, bbox_inches='tight')
plt.close()

all_x = np.vstack((x_opt[2:,0:2],x_opt[2:,2:4]))

refined_emitter_1 = np.array([p for p in x_opt[2:,0:2] if np.min([np.linalg.norm(i - p)for i in all_x[np.where(all_x != p)[0]]]) < 0.0015])
refined_emitter_2 = np.array([p for p in x_opt[2:,2:4] if np.min([np.linalg.norm(i - p)for i in all_x[np.where(all_x != p)[0]]]) < 0.0015])

plt.scatter(refined_emitter_1[:,0],refined_emitter_1[:,1], c='b', s=2., marker='.')
plt.scatter(refined_emitter_2[:,0],refined_emitter_2[:,1], c='r', s=2., marker='.')
plt.scatter(x_opt[0,0],x_opt[0,1], c='k', marker='x', s=40, linewidths=1.0)
plt.scatter(x_opt[0,2],x_opt[0,3], c='k', marker='x', s=40, linewidths=1.0)
plt.scatter(x_opt[1,0],x_opt[1,1], c='k', s=40, marker='+', linewidths=0.5)
plt.scatter(x_opt[1,2],x_opt[1,3], c='k', s=40, marker='+', linewidths=0.5)    
plt.xlim((plot_center[0],x_range[1]))
plt.ylim((-0.3 * emitter_separation + plot_center[1],plot_center[1]))
plt.xlabel(r"$x/w$", fontsize=18)
plt.ylabel(r"$y/w$", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().set_aspect(1)
plt.tight_layout()
plt.savefig('x_opt_quadrant_refined.png', dpi=600, bbox_inches='tight')
plt.close()

refined_emitter_2 = refined_emitter_2[list(set(np.where(refined_emitter_2[:,1] < -0.06)[0]).intersection(np.where(refined_emitter_2[:,0] > 0.0)[0]))]

km = KMeans(2, max_iter=300).fit(refined_emitter_2).cluster_centers_

plt.scatter(refined_emitter_2[:,0],refined_emitter_2[:,1], c='r', s=2., marker='.')
plt.scatter(km[:,0],km[:,1], c='k', marker='+', s=40, linewidths=1.0) 
plt.xlim((plot_center[0],x_range[1]))
plt.ylim((-0.3 * emitter_separation + plot_center[1],plot_center[1]))
plt.xlabel(r"$x/w$", fontsize=18)
plt.ylabel(r"$y/w$", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().set_aspect(1)
plt.tight_layout()
plt.savefig('x_opt_quadrant_refined_k_means.png', dpi=600, bbox_inches='tight')
plt.close()

np.savetxt('k_means.csv', km, delimiter=',')