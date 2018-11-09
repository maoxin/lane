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

    def __init__(self, extent_x, extent_y, one_point=True, **kwargs):
        """
        [1]N. K. Kanhere and S. T. Birchfield,
        “A Taxonomy and Analysis of Camera Calibration Methods for Traffic Monitoring Applications,”
        IEEE Transactions on Intelligent Transportation Systems, vol. 11, no. 2, pp. 441–452, Jun. 2010.

        upper left corner as origin point

        road direction: p1, p2 | p3, p4
        perpendicular direction: p1, p4 | p2, p3
        width x value: u2, u3,
        length y value: vb, vf
        width: w
        length: l
        """

        self.shift_x = extent_x / 2
        self.shift_y = extent_y / 2
        self.one_point = one_point

        self.p1 = kwargs['p1']
        self.p2 = kwargs['p2']
        self.p3 = kwargs['p3']
        self.p4 = kwargs['p4']

        self.u2 = kwargs['u2']
        self.u3 = kwargs['u3']

        self.p1 = (self.p1[0] - self.shift_x, self.p1[1] - self.shift_y)
        self.p2 = (self.p2[0] - self.shift_x, self.p2[1] - self.shift_y)
        self.p3 = (self.p3[0] - self.shift_x, self.p3[1] - self.shift_y)
        self.p4 = (self.p4[0] - self.shift_x, self.p4[1] - self.shift_y)
        self.u2 = self.u2 - self.shift_x
        self.u3 = self.u3 - self.shift_x

        self.w = kwargs['w']

        if self.one_point:
            self.vb = kwargs['vb']
            self.vf = kwargs['vf']
            
            self.vb = self.vb - self.shift_y
            self.vf = self.vf - self.shift_y

            self.l = kwargs['l']

        self.u0, self.v0 = self.__compute_1st_vanishing_points()
        if not self.one_point:
            self.u1 = self.__compute_2nd_vanishing_points()

    def __compute_1st_vanishing_points(self):
        road_l1 = Line(Point(self.p1), Point(self.p2))
        road_l2 = Line(Point(self.p3), Point(self.p4))

        road_intersection = intersection(road_l1, road_l2)
        u0 = float(road_intersection[0][0])
        v0 = float(road_intersection[0][1])

        return u0, v0

    def __compute_2nd_vanishing_points(self):
        # perp_l1 = Line(Point(self.p2), Point(self.p3))
        perp_l1 = Line(Point(self.p1), Point(self.p4))
        # horizon_l = Line(Point(0, self.v0), (1, self.v0))
        horizon_l = Line(Point(self.p2), Point(self.p3))

        perp_intersection = intersection(perp_l1, horizon_l)
        u1 = float(perp_intersection[0][0])

        return u1

    def calibration(self, save=True):
        delta = self.u3 - self.u2

        if not self.one_point:
            fs = [np.sqrt(-(self.v0**2 + self.u0 * self.u1))]
        else:
            k = (self.vf - self.v0) * (self.vb - self.v0) / (self.vf - self.vb)
            k_v = delta * k * self.l / (self.w * self.v0)

            B = 2 * (self.u0**2 + self.v0**2) - k_v**2
            C = (self.u0**2 + self.v0**2)**2 - k_v**2 * self.v0**2

            # print('delta', delta)
            # print('k', k)
            # print('k_v', k_v)
            # print('u0', self.u0)
            # print('v0', self.v0)
            # print('B', B)
            # print('C', C)

            f_square_0 = (-B + np.sqrt(B**2 - 4 * C)) / 2
            f_square_1 = (-B - np.sqrt(B**2 - 4 * C)) / 2

            fs = [np.sqrt(f_square_0)]
            if f_square_1 >= 0:
                fs.append(np.sqrt(f_square_1))

        phis = []
        thetas = []
        hs = []
        for f in fs:
            phi = np.arctan(-self.v0 / f)
            theta = np.arctan(-self.u0 * np.cos(phi) / f)
            h = f * self.w * np.sin(phi) / (np.abs(delta) * np.cos(theta))
            phis.append(phi)
            thetas.append(theta)
            hs.append(h)
            

        cc_parameter = {
            "f" : [float(f) for f in fs],
            "phi": [float(phi) for phi in phis],
            "theta": [float(theta) for theta in thetas],
            "h": [float(h) for h in hs],
        }

        if save:
            with open(os.path.join(dirname(realpath(__file__)),"cc_parameter.yaml"), 'w') as f_cc_parameter:
                f_cc_parameter.write(yaml.dump(cc_parameter))

        return cc_parameter
