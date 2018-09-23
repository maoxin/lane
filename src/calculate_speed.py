import pandas as pd
import numpy as np

from camera_calibration import CameraCalibration

import matplotlib.pyplot as plt

from os.path import join, realpath, dirname


class CalculateSpeed(object):
    def __init__(self, filename=join(dirname(dirname(realpath(__file__))), "data/cctv_ftg1_SPEED_000000_carLoc.csv"),
                 top_obs_rec=906, bottom_obs_rec=439,
                 window_size=10):
        self.start_time = []
        self.car_type = []
        self.track_coord = []
        with open(filename, 'r') as f:
            for line in f:
                dt = line.strip().split("; ")

                self.start_time.append(float(dt[0]) * 0.1)
                self.car_type.append(dt[1])
                self.track_coord.append([eval(x) for x in dt[2:]])
                
        self.start_time = np.array(self.start_time)
        self.car_type = np.array(self.car_type)

        self.top_obs_rec = top_obs_rec
        self.bottom_obs_rec = bottom_obs_rec

        self.window_size = window_size
        
        self.cc = CameraCalibration()

    def coord_image2word(self):
        self.Us = []
        self.Vs = []
        self.times = []
        for i, track in enumerate(self.track_coord):
            dt = np.dtype('float,float')
            track = np.array(track, dtype=dt)

            U = track['f0']
            V = track['f1']
            time = self.start_time[i] + np.arange(U.shape[0]) * 0.1

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
                    (X[self.window_size:] - X[:-self.window_size])**2 +
                    (Y[self.window_size:] - Y[:-self.window_size])**2
                ) / (time[self.window_size:] - time[:-self.window_size]) * 3.6
            )
            # m/s -> km/h

        return self.speeds

    def save_result(self):
        df = pd.DataFrame(columns=['time', 'speed', 'car_type'])
        
        all_speeds = np.hstack(self.speeds)
        all_times = [(tm[self.window_size:] + tm[:-self.window_size]) / 2 for tm in self.times]
        all_times = np.hstack(all_times).astype('timedelta64[s]') + np.datetime64('2018-01-01T08:00')

        df.speed = all_speeds
        df.time = all_times
        i = 0
        for j in range(len(self.speeds)):
            df.loc[i: i+self.speeds[j].shape[0], 'car_type'] = self.car_type[j]
            i = i + self.speeds[j].shape[0]

        df.to_csv(join(dirname(dirname(realpath(__file__))), 'result/speeds.csv'), index=False)

        return 0


    def draw_result(self):
        for i, (car_type, speed, time) in enumerate(zip(self.car_type, self.speeds, self.times)):
            plt.close()
            plt.plot((time[self.window_size:] + time[:-self.window_size]) / 2, speed, 'o-')
            plt.title(f"{car_type} speed")
            plt.xlabel('time (s)')
            plt.ylabel('speed (km/h)')
            plt.savefig(join(dirname(dirname(realpath(__file__))), f'result/speed/{i}.pdf'))
            plt.close()

        for i, (car_type, X, Y) in enumerate(zip(self.car_type, self.Xs, self.Ys)):
            plt.close()
            plt.plot(X, Y)
            plt.title(f"{car_type} track")
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.savefig(join(dirname(dirname(realpath(__file__))), f'result/track/{i}.pdf'))
            plt.close()

        for i, (car_type, u, v) in enumerate(zip(self.car_type, self.Us, self.Vs)):
            plt.close()
            plt.plot(u, v)
            plt.title(f"{car_type} track")
            plt.xlabel('U (pixel)')
            plt.ylabel('V (pixel)')
            plt.savefig(join(dirname(dirname(realpath(__file__))),f'result/track_image/{i}.pdf'))
            plt.close()

        return 0


if __name__ == "__main__":
    cs = CalculateSpeed()
    speeds = cs.get_speed()
    # cs.draw_result()
    cs.save_result()
