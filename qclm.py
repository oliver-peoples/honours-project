from dataclasses import dataclass
import numpy as np
from scipy.special import genlaguerre as genLaguerre
from scipy.special import hermite as genHermite
from scipy.special._orthogonal import orthopoly1d
from typing import List, Any


@dataclass
class GaussLaguerre:
    
    p: int
    l: int
    
    I_0: float
    waist: float
    
    center: np.ndarray = np.array([0.,0.])
    
    rotation: float = 0
    
    volume: float = 1.
    
    def __post_init__(self) -> None:
        
        self.laguerre = genLaguerre(self.p, self.l)
        
        grid_x = 3750
        grid_y = 3750
        
        x_linspace = np.linspace(-10, 10, grid_x)
        y_linspace = np.linspace(-10, 10, grid_y)
        
        x_meshgrid, y_meshgrid = np.meshgrid(
            x_linspace,
            y_linspace
        )
        
        intensity_map = self.intensityMap(x_meshgrid, y_meshgrid)
        
        volume = np.trapz(
            y=np.asarray(
                [np.trapz(y=intensity_row, x=x_linspace) for intensity_row in intensity_map[:]]
            ),
            x=y_linspace
        )
        
        self.volume = volume
        
    def orderString(self) -> str:
        
        return str(self.p) + str(self.l)
    
    def modeTypeString(self) -> str:
        
        return 'GL'
        
    def rhoFn(self, r) -> float:
    
        return 2. * r**2 / self.waist**2
        
    def intensityFn(self, x: float, y: float):
        
        relative_x = x - self.center[0]
        relative_y = y - self.center[1]
        
        r = np.sqrt(relative_x**2 + relative_y**2)
        phi = np.arctan2(relative_y, relative_x)
        phi += self.rotation
        
        rho = self.rhoFn(r)
        
        gl_tem = self.I_0 * (rho**self.l) * (self.laguerre(rho)**2) * (np.cos(self.l * phi)**2) * np.exp(-rho)
        
        return gl_tem / self.volume
    
    def intensityMap(self, x_meshgrid, y_meshgrid):
        
        return self.intensityFn(x_meshgrid, y_meshgrid)
    
@dataclass
class GaussHermite:
    
    m: int
    n: int
    
    I_0: float
    waist: float
    
    center: np.ndarray = np.array([0.,0.])
    
    rotation: float = 0.
    
    volume: float = 1.
    
    def __post_init__(self) -> None:
        
        self.gh_m = genHermite(self.m)
        self.gh_n = genHermite(self.n)
        
        grid_x = 3750
        grid_y = 3750
        
        x_linspace = np.linspace(-10, 10, grid_x)
        y_linspace = np.linspace(-10, 10, grid_y)
        
        x_meshgrid, y_meshgrid = np.meshgrid(
            x_linspace,
            y_linspace
        )
        
        intensity_map = self.intensityMap(x_meshgrid, y_meshgrid)
        
        volume = np.trapz(
            y=np.asarray(
                [np.trapz(y=intensity_row, x=x_linspace) for intensity_row in intensity_map[:]]
            ),
            x=y_linspace
        )
        
        self.volume = volume
        
    def orderString(self) -> str:
        
        return str(self.m) + str(self.n)
    
    def modeTypeString(self) -> str:
        
        return 'GH'
    
    def intensityFn(self, x_: float, y_: float):
        
        relative_x = x_ - self.center[0]
        relative_y = y_ - self.center[1]
        
        r = np.sqrt(relative_x**2 + relative_y**2)
        phi = np.arctan2(relative_y, relative_x)
        phi += self.rotation
        
        relative_x = np.cos(phi) * r
        relative_y = np.sin(phi) * r
        
        norm_coeff = 1. / (2. * self.waist**2 * np.pi * [0.5, 1, 4, 24][self.m] * [0.5, 1, 4, 24][self.n])
        
        return (1. / self.volume) * norm_coeff * (self.gh_m((2**0.5 * relative_x) / self.waist) * np.exp(-relative_x**2 / self.waist**2))**2 * (self.gh_n((2**0.5 * relative_y) / self.waist) * np.exp(-relative_y**2 / self.waist**2))**2
    
    def intensityMap(self, x_meshgrid, y_meshgrid):
        
        return self.intensityFn(x_meshgrid, y_meshgrid)
        
    
@dataclass
class Detector:
    
    waist = 1.
    
    center: np.ndarray = np.array([0.,0.])
    
    def detectFn(self, x: float, y: float, p: float):
        
        relative_x = x - self.center[0]
        relative_y = y - self.center[1]
        
        r = np.sqrt(relative_x**2 + relative_y**2)
        
        return np.exp(-(r**2 / 2) / (2 * self.waist**2)) * p

@dataclass
class Emitter:
    
    xy: np.array
    relative_brightness: float
    
@dataclass
class Solver:
    
    illumination_structures: List[Any]
    
    detector: Detector
    
    g_1_true: np.ndarray
    g_2_true: np.ndarray
    
    def __post_init__(self):
        
        self.optimization_lambda = lambda guess: self.rss(guess)
        
    def rss(self, guess):
        
        g_1_guess = np.zeros_like(self.g_1_true)
        g_2_guess = np.zeros_like(self.g_2_true)
        
        e_1_guess = Emitter(
            guess[:2],
            1.
        )
        
        e_2_guess = Emitter(
            guess[2:4],
            guess[4]
        )
        
        for is_idx in range(0,len(self.illumination_structures)):
    
            p_1_guess = e_1_guess.relative_brightness * (self.illumination_structures[is_idx].intensityFn(*e_1_guess.xy))
            p_2_guess = e_2_guess.relative_brightness * (self.illumination_structures[is_idx].intensityFn(*e_2_guess.xy))
            
            p_1_guess = self.detector.detectFn(*e_1_guess.xy, p_1_guess)
            p_2_guess = self.detector.detectFn(*e_2_guess.xy, p_2_guess)
            
            g_1_guess[is_idx] = (p_1_guess + p_2_guess) / (e_1_guess.relative_brightness + e_2_guess.relative_brightness)
            
            alpha = p_2_guess / p_1_guess
            
            g_2_guess[is_idx] = (2 * alpha) / (1 + alpha)**2
            
        return np.sum((self.g_1_true - g_1_guess)**2) + np.sum((self.g_2_true - g_2_guess)**2)

import numpy as np

class Point:
    """A point located at (x,y) in 2D space.

    Each Point object may be associated with a payload object.

    """

    def __init__(self, x, y, payload=None):
        self.x, self.y = x, y
        self.payload = payload

    def __repr__(self):
        return '{}: {}'.format(str((self.x, self.y)), repr(self.payload))
    def __str__(self):
        return 'P({:.2f}, {:.2f})'.format(self.x, self.y)

    def distance_to(self, other):
        try:
            other_x, other_y = other.x, other.y
        except AttributeError:
            other_x, other_y = other
        return np.hypot(self.x - other_x, self.y - other_y)

class Rect:
    """A rectangle centred at (cx, cy) with width w and height h."""

    def __init__(self, cx, cy, w, h):
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h
        self.west_edge, self.east_edge = cx - w/2, cx + w/2
        self.north_edge, self.south_edge = cy - h/2, cy + h/2

    def __repr__(self):
        return str((self.west_edge, self.east_edge, self.north_edge,
                self.south_edge))

    def __str__(self):
        return '({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(self.west_edge,
                    self.north_edge, self.east_edge, self.south_edge)

    def contains(self, point):
        """Is point (a Point object or (x,y) tuple) inside this Rect?"""

        try:
            point_x, point_y = point.x, point.y
        except AttributeError:
            point_x, point_y = point

        return (point_x >= self.west_edge and
                point_x <  self.east_edge and
                point_y >= self.north_edge and
                point_y < self.south_edge)

    def intersects(self, other):
        """Does Rect object other interesect this Rect?"""
        return not (other.west_edge > self.east_edge or
                    other.east_edge < self.west_edge or
                    other.north_edge > self.south_edge or
                    other.south_edge < self.north_edge)

    def draw(self, ax, c='k', lw=0.25, **kwargs):
        x1, y1 = self.west_edge, self.north_edge
        x2, y2 = self.east_edge, self.south_edge
        ax.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], c=c, lw=lw, **kwargs)


class QuadTree:
    """A class implementing a quadtree."""

    def __init__(self, boundary, max_points=4, depth=0):
        """Initialize this node of the quadtree.

        boundary is a Rect object defining the region from which points are
        placed into this node; max_points is the maximum number of points the
        node can hold before it must divide (branch into four more nodes);
        depth keeps track of how deep into the quadtree this node lies.

        """

        self.boundary = boundary
        self.max_points = max_points
        self.points = []
        self.depth = depth
        # A flag to indicate whether this node has divided (branched) or not.
        self.divided = False

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = ' ' * self.depth * 2
        s = str(self.boundary) + '\n'
        s += sp + ', '.join(str(point) for point in self.points)
        if not self.divided:
            return s
        return s + '\n' + '\n'.join([
                sp + 'nw: ' + str(self.nw), sp + 'ne: ' + str(self.ne),
                sp + 'se: ' + str(self.se), sp + 'sw: ' + str(self.sw)])

    def divide(self):
        """Divide (branch) this node by spawning four children nodes."""

        cx, cy = self.boundary.cx, self.boundary.cy
        w, h = self.boundary.w / 2, self.boundary.h / 2
        # The boundaries of the four children nodes are "northwest",
        # "northeast", "southeast" and "southwest" quadrants within the
        # boundary of the current node.
        self.nw = QuadTree(Rect(cx - w/2, cy - h/2, w, h),
                                    self.max_points, self.depth + 1)
        self.ne = QuadTree(Rect(cx + w/2, cy - h/2, w, h),
                                    self.max_points, self.depth + 1)
        self.se = QuadTree(Rect(cx + w/2, cy + h/2, w, h),
                                    self.max_points, self.depth + 1)
        self.sw = QuadTree(Rect(cx - w/2, cy + h/2, w, h),
                                    self.max_points, self.depth + 1)
        self.divided = True

    def insert(self, point):
        """Try to insert Point point into this QuadTree."""

        if not self.boundary.contains(point):
            # The point does not lie inside boundary: bail.
            return False
        if len(self.points) < self.max_points:
            # There's room for our point without dividing the QuadTree.
            self.points.append(point)
            return True

        # No room: divide if necessary, then try the sub-quads.
        if not self.divided:
            self.divide()

        return (self.ne.insert(point) or
                self.nw.insert(point) or
                self.se.insert(point) or
                self.sw.insert(point))

    def query(self, boundary, found_points):
        """Find the points in the quadtree that lie within boundary."""

        if not self.boundary.intersects(boundary):
            # If the domain of this node does not intersect the search
            # region, we don't need to look in it for points.
            return False

        # Search this node's points to see if they lie within boundary ...
        for point in self.points:
            if boundary.contains(point):
                found_points.append(point)
        # ... and if this node has children, search them too.
        if self.divided:
            self.nw.query(boundary, found_points)
            self.ne.query(boundary, found_points)
            self.se.query(boundary, found_points)
            self.sw.query(boundary, found_points)
        return found_points


    def query_circle(self, boundary, centre, radius, found_points):
        """Find the points in the quadtree that lie within radius of centre.

        boundary is a Rect object (a square) that bounds the search circle.
        There is no need to call this method directly: use query_radius.

        """

        if not self.boundary.intersects(boundary):
            # If the domain of this node does not intersect the search
            # region, we don't need to look in it for points.
            return False

        # Search this node's points to see if they lie within boundary
        # and also lie within a circle of given radius around the centre point.
        for point in self.points:
            if (boundary.contains(point) and
                    point.distance_to(centre) <= radius):
                found_points.append(point)

        # Recurse the search into this node's children.
        if self.divided:
            self.nw.query_circle(boundary, centre, radius, found_points)
            self.ne.query_circle(boundary, centre, radius, found_points)
            self.se.query_circle(boundary, centre, radius, found_points)
            self.sw.query_circle(boundary, centre, radius, found_points)
        return found_points

    def query_radius(self, centre, radius, found_points):
        """Find the points in the quadtree that lie within radius of centre."""

        # First find the square that bounds the search circle as a Rect object.
        boundary = Rect(*centre, 2*radius, 2*radius)
        return self.query_circle(boundary, centre, radius, found_points)


    def __len__(self):
        """Return the number of points in the quadtree."""

        npoints = len(self.points)
        if self.divided:
            npoints += len(self.nw)+len(self.ne)+len(self.se)+len(self.sw)
        return npoints

    def draw(self, ax):
        """Draw a representation of the quadtree on Matplotlib Axes ax."""

        self.boundary.draw(ax)
        if self.divided:
            self.nw.draw(ax)
            self.ne.draw(ax)
            self.se.draw(ax)
            self.sw.draw(ax)
