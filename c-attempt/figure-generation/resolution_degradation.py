# from scipy.special import hermite as physicistsHermite
import numpy as np
# from parula import parula
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import RegularPolygon, Circle
import os
import cv2

# from scipy.spatial import ConvexHull, convex_hull_plot_2d

cm = 1/2.54

path = os.path.dirname(__file__)

matplotlib.rcParams['text.usetex'] = True

rmit_logo = cv2.imread(os.path.join(path,'rmit_logo.png'))

rmit_logo_blurred = cv2.GaussianBlur(rmit_logo,(111,111),sigmaX=10.)

cv2.imwrite(os.path.join(path,'rmit_logo_blurred.png'), rmit_logo_blurred)