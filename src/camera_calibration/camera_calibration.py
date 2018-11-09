import numpy as np
import sys
# from camera_calibration.parameter_compute import ParameterCompute
from parameter_compute import ParameterCompute
import yaml
from os.path import join, realpath, dirname
# from camera_calibration.undistortion import UnRadialDistortion
from undistortion import UnRadialDistortion

class CameraCalibration(object):
    """
    Schoepflin, T.N., and D.J. Dailey. 2003.
    “Dynamic Camera Calibration of Roadside Traffic Management Cameras for Vehicle Speed Estimation.”
    IEEE Transactions on Intelligent Transportation Systems 4 (2): 90–98. 
    https://doi.org/10.1109/TITS.2003.821213.
    """
    def __init__(self, extent_x=1920, extent_y=1080, one_point=True,
                 **kwargs):
        """
        h: height (m)
        phi: tilt (˚)
        f: focal length (pixel)
        v0: ordinate of the first vanishing point in the paper (pixel)
        """

        self.undistort = UnRadialDistortion()
        kwargs['p1'] = self.undistort(kwargs['p1'])
        kwargs['p2'] = self.undistort(kwargs['p2'])
        kwargs['p3'] = self.undistort(kwargs['p3'])
        kwargs['p4'] = self.undistort(kwargs['p4'])
        # kwargs['p5'] = self.undistort(kwargs['p5'])
        # kwargs['p6'] = self.undistort(kwargs['p6'])
        # kwargs['p7'] = self.undistort(kwargs['p7'])
        # kwargs['p8'] = self.undistort(kwargs['p8'])
        # kwargs['u2'] = self.undistort((kwargs['u2'], 540))[0]
        # kwargs['u3'] = self.undistort((kwargs['u3'], 540))[0]
        if one_point:
            kwargs['vf'] = self.undistort(kwargs['pvf'])[1]
            kwargs['vb'] = self.undistort(kwargs['pvb'])[1]

        pc = ParameterCompute(extent_x, extent_y, one_point, **kwargs)

        cc_parameter = pc.calibration()

        self.f = cc_parameter['f'][0]
        self.phi = cc_parameter['phi'][0]
        self.h = cc_parameter['h'][0]

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
        u, v = self.undistort((u, v))

        u = u - self.shift_x
        v = v - self.shift_y
        
        X = self.h * u / np.cos(self.phi) / (v + self.f * np.tan(self.phi))
        Y = self.h * (self.f - v * np.tan(self.phi)) / (v + self.f * np.tan(self.phi))

        return (X, Y)

if __name__ == "__main__":
    parameter0 = {
        'p1': (947.638, 767.447),
        'p2': (709.228, 514.002), 
        'p3': (806.093, 504.293),
        'p4': (1089.306, 736.378),
        # 'p5': (190.549, 489.880),
        # 'p6': (711.208, 428.578),
        # 'p7': (255.486, 394.890),
        # 'p8': (625.018, 344.884),
        'u2': 728.476, 'u3': 844.025,
        'w': 2.55,
    }

    parameter1 = {
        'p1': (787.919, 640.099),
        'p2': (594.782, 404.630),
        'p3': (675.065, 399.513),
        'p4': (913.843, 619.450),
        # 'p5': (457.800, 335.916),
        # 'p6': (603.722, 326.745),
        # 'p7': (429.050, 256.697),
        # 'p8': (533.523, 252.352),

        'u2': 703.475, 'u3': 823.241,
        'pvb': (612.170, 458.901),
        'pvf': (885.514, 765.706),
        'w': 2.55,
        'l': 11.0,
    }

    cc0 = CameraCalibration(one_point=False, **parameter0)
    cc1 = CameraCalibration(one_point=True, **parameter1)
    x00, y00 = cc0.image2world(634.605, 439.723)
    x01, y01 = cc0.image2world(1088.462, 931.761)

    x10, y10 = cc1.image2world(538.095, 337.152)
    x11, y11 = cc1.image2world(907.756, 793.968)

    print('case0:', np.sqrt((x01 - x00)**2 + (y01 - y00)**2)) # 31.05945702896952
    print('case1:', np.sqrt((x11 - x10)**2 + (y11 - y10)**2)) # 51.07394876776243
