import sys
from os.path import abspath, dirname, realpath, join
import numpy as np

sys.path.append(abspath(join(dirname(dirname(realpath(__file__))), 'src')))

from camera_calibration import CameraCalibration

class TestCameraCalibration(object):
    def test_car_length(self):
        cc = CameraCalibration()

        X1, Y1 = cc.image2world(661.498, 465.096)
        X2, Y2 = cc.image2world(799.19, 617.622)

        length = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2)

        real_length = 11

        assert np.abs(length - real_length) < 4

