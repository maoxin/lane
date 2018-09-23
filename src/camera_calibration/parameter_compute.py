from sympy import Line, Point, intersection
import numpy as np
import yaml
import os.path
from os.path import realpath, dirname

class ParameterCompute(object):
    """
    Schoepflin, T.N., and D.J. Dailey. 2003.
    “Dynamic Camera Calibration of Roadside Traffic Management Cameras for Vehicle Speed Estimation.”
    IEEE Transactions on Intelligent Transportation Systems 4 (2): 90–98. 
    https://doi.org/10.1109/TITS.2003.821213.
    """

    def __init__(self, extent_x, extent_y, p1, p2, p3, p4, u2, u4, w):
        """
        upper left corner as origin point

        road direction: p1, p2 | p3, p4
        perpendicular direction: p1, p4 | p2, p3
        """

        self.shift_x = extent_x / 2
        self.shift_y = extent_y / 2

        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

        self.u2 = u2
        self.u4 = u4

        self.w = w


    def compute_vanishing_points(self):
        road_l1 = Line(Point(self.p1), Point(self.p2))
        road_l2 = Line(Point(self.p3), Point(self.p4))

        road_intersection = intersection(road_l1, road_l2)
        u0 = float(road_intersection[0][0])
        v0 = float(road_intersection[0][1])

        perp_l1 = Line(Point(self.p2), Point(self.p3))
        horizon_l = Line(Point(0, v0), (1, v0))

        perp_intersection = intersection(perp_l1, horizon_l)
        u1 = float(perp_intersection[0][0])

        u0 = u0 - self.shift_x
        u1 = u1 - self.shift_x
        v0 = v0 - self.shift_y
        # set center as origin point

        return u0, v0, u1

    def calibration(self, save=True):
        u0, v0, u1 = self.compute_vanishing_points()
        f = np.sqrt(-(v0**2 + u0 * u1))
        phi = np.arctan(-v0 / f)
        theta = np.arctan(-u0 * np.cos(phi) / f)
        h = f * self.w * np.sin(phi) / (np.abs(self.u4 - self.u2) * np.cos(theta))

        cc_parameter = {
            "f" : float(f),
            "phi": float(phi),
            "theta": float(theta),
            "h": float(h),
        }

        if save:
            with open(os.path.join(dirname(realpath(__file__)),"cc_parameter.yaml"), 'w') as f_cc_parameter:
                f_cc_parameter.write(yaml.dump(cc_parameter))

        return cc_parameter
