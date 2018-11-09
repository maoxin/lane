import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import cv2

class UnRadialDistortion():
    def __init__(self, points=[(200.567, 1061.806), (193.522, 964.531), (192.431, 926.184),
                (192.931, 883.980), (192.431, 848.80), (194.620, 806.819), (197.564, 766.16),
                (201.599, 731.305), (204.969, 697.170), (207.905, 669.216), (212.628, 639.976),
                (217.922, 607.673), (224.142, 580.360), (231.289, 552.760), (237.505, 517.240),
                (245.825, 485.787), (255.099, 450.621), (263.067, 422.758), (271.269, 394.379),
                (279.898, 368.462)] , img_width=1920, img_height=1080):
        self.img_width = img_width
        self.img_height = img_height

        self.points = points
        self.xs = np.array([i[0] - self.img_width / 2 for i in self.points]) / self.img_width
        self.ys = np.array([i[1] - self.img_height / 2 for i in self.points]) / self.img_height
        # normalization is needed

        self.rs = np.sqrt(self.xs**2 + self.ys**2)
        self.k1s = np.linspace(-1, 1, 1000)
        self.k2s = np.linspace(-1, 1, 1000)

        self.k1s, self.k2s = np.meshgrid(self.k1s, self.k2s)

        self.k1, self.k2, self.stds = self.__get_coeff_distortion()

    def __call__(self, xy):
        x_normal = (xy[0] - self.img_width / 2) / self.img_width
        y_normal = (xy[1] - self.img_height / 2) / self.img_height
        r_normal = np.sqrt(x_normal**2 + y_normal**2)

        x_new = x_normal * (1 + self.k1 * r_normal**2 + self.k2 * r_normal**4)
        y_new = y_normal * (1 + self.k1 * r_normal**2 + self.k2 * r_normal**4)

        x_new = x_new * self.img_width + self.img_width / 2
        y_new = y_new * self.img_height + self.img_height / 2

        return (x_new, y_new)
        # return (xy[0], xy[1])

    def __get_coeff_distortion(self):
        xs_cr = self.xs[:, None, None] * (1 + self.k1s * (self.rs**2)[:, None, None] + self.k2s * (self.rs**4)[:, None, None])
        ys_cr = self.ys[:, None, None] * (1 + self.k1s * (self.rs**2)[:, None, None] + self.k2s * (self.rs**4)[:, None, None])

        ks = (ys_cr[1:] - ys_cr[:-1]) / (xs_cr[1:] - xs_cr[:-1])
        stds = np.sqrt( ((ks - ks.mean(axis=0))**2).mean(axis=0) )

        ind_coef = np.unravel_index(np.argmin(stds), (1000, 1000))

        # plt.imshow(np.log(stds), cmap='hsv')
        # plt.show()

        return self.k1s[ind_coef], self.k2s[ind_coef], stds

    def undistort_ar_img(self, ar_img):
        x, y = np.meshgrid(range(self.img_width), range(self.img_height))
        x_normal = (x - self.img_width / 2) / self.img_width
        y_normal = (y - self.img_height / 2) / self.img_height

        r_normal = np.sqrt(x_normal**2 + y_normal**2)
        x_new = ((1 + self.k1 * r_normal**2 + self.k2 * r_normal**4) * x_normal * self.img_width + self.img_width/2).astype(int)
        y_new = ((1 + self.k1 * r_normal**2 + self.k2 * r_normal**4) * y_normal * self.img_height + self.img_height/2).astype(int)

        x = x.flatten()
        y = y.flatten()
        x_new = x_new.flatten()
        y_new = y_new.flatten()

        ind2use = (x_new >= 0) & (x_new < self.img_width) & (y_new >= 0) & (y_new < self.img_height)

        x = x[ind2use]
        y = y[ind2use]
        x_new = x_new[ind2use]
        y_new = y_new[ind2use]

        new_ar_img = np.zeros(ar_img.shape).astype('uint8')
        new_ar_img[y_new, x_new] = ar_img[y, x]

        return new_ar_img





if __name__ == '__main__':
    points = [
        (200.567, 1061.806),
        (193.522, 964.531),
        (192.431, 926.184),
        (192.931, 883.980),
        (192.431, 848.80),
        (194.620, 806.819),
        (197.564, 766.16),
        (201.599, 731.305),
        (204.969, 697.170),
        (207.905, 669.216),
        (212.628, 639.976),
        (217.922, 607.673),
        (224.142, 580.360),
        (231.289, 552.760),
        (237.505, 517.240),
        (245.825, 485.787),
        (255.099, 450.621),
        (263.067, 422.758),
        (271.269, 394.379),
        (279.898, 368.462)
    ]

    urd = UnRadialDistortion(points=points, img_width=1920, img_height=1072)
    ar_img = cv2.imread('/Users/maoxin/Desktop/cctv_ftg6-0003.png')
    new_ar_img = urd.undistort_ar_img(ar_img)
