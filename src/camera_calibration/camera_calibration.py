import numpy as np
import sys
from camera_calibration.parameter_compute import ParameterCompute

class CameraCalibration(object):
    """
    Schoepflin, T.N., and D.J. Dailey. 2003.
    “Dynamic Camera Calibration of Roadside Traffic Management Cameras for Vehicle Speed Estimation.”
    IEEE Transactions on Intelligent Transportation Systems 4 (2): 90–98. 
    https://doi.org/10.1109/TITS.2003.821213.
    """
    def __init__(self, extent_x=1920, extent_y=1080,
                 p1=(947.638, 767.447), p2=(709.228, 514.002), 
                 p3=(806.093, 504.293), p4=(1089.306, 736.378),
                 u2=728.476, u4=844.025, w=2.55):
        """
        h: height (m)
        phi: tilt (˚)
        f: focal length (pixel)
        v0: ordinate of the first vanishing point in the paper (pixel)
        """  

        pc = ParameterCompute(extent_x, extent_y,
                                p1, p2, p3, p4,
                                u2, u4, w)
        cc_parameter = pc.calibration()

        self.f = cc_parameter['f']
        self.phi = cc_parameter['phi']
        self.h = cc_parameter['h']

        self.shift_x = extent_x / 2
        self.shift_y = extent_y / 2

    
    def image2world(self, u, v):
        """
        u: abscissa of point in image (pixel)
        v: ordinate of point in image (pixel)
        
        X: abscissa of point in real world (m)
        y: ordinate of point in real world (m)

        numpy array allowed
        """
        u = u - self.shift_x
        v = v - self.shift_y
        
        X = self.h * u / np.cos(self.phi) / (v + self.f * np.tan(self.phi))
        Y = self.h * (self.f - v * np.tan(self.phi)) / (v + self.f * np.tan(self.phi))

        return (X, Y)
