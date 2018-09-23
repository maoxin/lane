import numpy as np

class CameraCalibration(object):
    """
    Schoepflin, T.N., and D.J. Dailey. 2003.
    “Dynamic Camera Calibration of Roadside Traffic Management Cameras for Vehicle Speed Estimation.”
    IEEE Transactions on Intelligent Transportation Systems 4 (2): 90–98. 
    https://doi.org/10.1109/TITS.2003.821213.
    """
    def __init__(self, frameX=1920, frameY=1080, h=9.512500591, phi=16.1017668, f=1436.887782):
        """
        h: height (m)
        phi: tilt (˚)
        f: focal length (pixel)
        v0: ordinate of the first vanishing point in the paper (pixel)
        """
        self.h = h
        self.phi = np.deg2rad(phi)
        self.f = f

        self.shiftX = frameX // 2
        self.shiftY = frameY // 2

    
    def image2world(self, u, v):
        """
        u: abscissa of point in image (pixel)
        v: ordinate of point in image (pixel)
        
        X: abscissa of point in real world (m)
        y: ordinate of point in real world (m)
        """
        u = u - self.shiftX
        v = v - self.shiftY
        # X = self.S * (u * self.v0) / (self.v0 - v)
        # Y = self.S / np.sin(self.phi) * (v * self.v0) / (self.v0 - v)
        
        X = self.h * u / np.cos(self.phi) / (v + self.f * np.tan(self.phi))
        Y = self.h * (self.f - v * np.tan(self.phi)) / (v + self.f * np.tan(self.phi))

        return (X, Y)


if __name__ == "__main__":
    cc = CameraCalibration()
    # X1, Y1 = cc.image2world(1920, 1080)
    # X2, Y2 = cc.image2world(0, 0)

    # X1, Y1 = cc.image2world(717.122, 588.299)
    # X2, Y2 = cc.image2world(1139.356, 992.978)
    # car length

    # X1, Y1 = cc.image2world(1168.892, 990.808)
    # X2, Y2 = cc.image2world(1301.005, 929.455)
    # car width

    # X1, Y1 = cc.image2world(430.417, 173.007)
    # X2, Y2 = cc.image2world(1242.996, 1075.958)
    # road length

    # X1, Y1 = cc.image2world(371.057, 159.215)
    # X2, Y2 = cc.image2world(332.656, 1068.205)
    # road length2

    # X1, Y1 = cc.image2world(483.667, 168.390)
    # X2, Y2 = cc.image2world(1630.960, 941.156)
    # road length3

    X1, Y1 = cc.image2world(661.498, 465.096)
    X2, Y2 = cc.image2world(799.19, 617.622)
    # car length


    print(X1, Y1)
    print(X2, Y2)
    print(np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2))
