import pandas as pd
import numpy as np
from camera_calibration import CameraCalibration
import matplotlib.pyplot as plt

class CalculateSpeed(object):
    def __init__(self, filename="./cctv_ftg1_000000_carLoc.csv"):
        self.df = pd.read_csv(filename, delimiter="; ", header=None)
        self.car_type = self.df.loc[:, 1]
        self.track_coord = self.df.loc[:, 2:]
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
            time = np.arange(U.shape[0]) * 0.1

            idx = ((V >= 439) & (V <= 906))
            U = U[idx]
            V = V[idx]
            time = time[idx]

            # U = U[::10]
            # V = V[::10]
            # time = time[::10]

            # length = U.shape[0]
            # U = U[length * 2 // 3: length * 3 // 4]
            # V = V[length * 2 // 3: length * 3 // 4]

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
        # km/h

        Xs, Ys, times = self.coord_image2word()
        self.speeds = []
        for X, Y, time in zip(Xs, Ys, times):
            self.speeds.append(np.sqrt((X[10:] - X[:-10])**2 + (Y[10:] - Y[:-10])**2) / (time[10:] - time[:-10]) * 3.6)

        return self.speeds

    def draw(self):
        for i, (car_type, speed, time) in enumerate(zip(self.car_type, self.speeds, self.times)):
            plt.close()
            plt.plot((time[10:] + time[:-10]) / 2, speed, 'o-')
            plt.title(f"{car_type} speed")
            plt.xlabel('time (s)')
            plt.ylabel('speed (km/h)')
            plt.savefig(f'./speed/{i}.pdf')
            plt.close()

        for i, (car_type, X, Y) in enumerate(zip(self.car_type, self.Xs, self.Ys)):
            plt.close()
            plt.plot(X, Y)
            plt.title(f"{car_type} track")
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.savefig(f'./track/{i}.pdf')
            plt.close()

        for i, (car_type, u, v) in enumerate(zip(self.car_type, self.Us, self.Vs)):
            plt.close()
            plt.plot(u, v)
            plt.title(f"{car_type} track")
            plt.xlabel('U (pixel)')
            plt.ylabel('V (pixel)')
            plt.savefig(f'./track_image/{i}.pdf')
            plt.close()

        return 0

            

if __name__ == "__main__":
    cs = CalculateSpeed()
    speeds = cs.get_speed()
    cs.draw()
