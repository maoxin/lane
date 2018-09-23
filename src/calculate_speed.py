import pandas as pd
import numpy as np

from camera_calibration import CameraCalibration

import matplotlib.pyplot as plt


class CalculateSpeed(object):
    def __init__(self, filename="../data/cctv_ftg1_000000_carLoc.csv",
                 top_obs_rec=906, bottom_obs_rec=439,
                 window_size=10):
        self.df = pd.read_csv(filename, delimiter="; ", header=None)
        self.start_time = self.df.loc[:, 0] * 0.1
        self.car_type = self.df.loc[:, 1]
        self.track_coord = self.df.loc[:, 2:]

        self.top_obs_rec = top_obs_rec
        self.bottom_obs_rec = bottom_obs_rec

        self.window_size = window_size
        
        self.cc = CameraCalibration()

    def coord_image2word(self):
        self.Us = []
        self.Vs = []
        self.times = []
        for i, track in self.track_coord.iterrows():
            track = track.dropna().apply(eval)
            dt = np.dtype('float,float')
            track = np.array(track, dtype=dt)

            U = track['f0']
            V = track['f1']
            time = self.start_time.loc[i] + np.arange(U.shape[0]) * 0.1

            idx = ((V >= self.bottom_obs_rec) & (V <= self.top_obs_rec))
            U = U[idx]
            V = V[idx]
            time = time[idx]

            self.Us.append(U)
            self.Vs.append(V)
            self.times.append(time)

        self.Xs = []
        self.Ys = []
        for U, V in zip(self.Us, self.Vs):
            X, Y = self.cc.image2world(U, V)
            self.Xs.append(X)
            self.Ys.append(Y)

        return self.Xs, self.Ys, self.times
    
    def get_speed(self):
        Xs, Ys, times = self.coord_image2word()
        self.speeds = []
        for X, Y, time in zip(Xs, Ys, times):
            self.speeds.append(
                np.sqrt(
                    (
                    X[self.window_size:] - X[:-self.window_size])**2 +
                    (Y[self.window_size:] - Y[:-self.window_size])**2
                    ) /
                    (time[self.window_size:] - time[:-self.window_size]) * 3.6
                )
            # m/s -> km/h

        return self.speeds

    def draw_result(self):
        for i, (car_type, speed, time) in enumerate(zip(self.car_type, self.speeds, self.times)):
            plt.close()
            plt.plot((time[self.window_size:] + time[:-self.window_size]) / 2, speed, 'o-')
            plt.title(f"{car_type} speed")
            plt.xlabel('time (s)')
            plt.ylabel('speed (km/h)')
            plt.savefig(f'../result/speed/{i}.pdf')
            plt.close()

        for i, (car_type, X, Y) in enumerate(zip(self.car_type, self.Xs, self.Ys)):
            plt.close()
            plt.plot(X, Y)
            plt.title(f"{car_type} track")
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.savefig(f'../result/track/{i}.pdf')
            plt.close()

        for i, (car_type, u, v) in enumerate(zip(self.car_type, self.Us, self.Vs)):
            plt.close()
            plt.plot(u, v)
            plt.title(f"{car_type} track")
            plt.xlabel('U (pixel)')
            plt.ylabel('V (pixel)')
            plt.savefig(f'../result/track_image/{i}.pdf')
            plt.close()

        return 0

            

if __name__ == "__main__":
    cs = CalculateSpeed()
    speeds = cs.get_speed()
    cs.draw_result()
