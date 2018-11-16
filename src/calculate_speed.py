import pandas as pd
import numpy as np
# from camera_calibration import CameraCalibration
import matplotlib.pyplot as plt
from os.path import join, realpath, dirname
from utils.handle_data_file import DataFileReader
from running_ruler.running_ruler import RunningRuler

class CalculateSpeed(object):
    data = DataFileReader(fp_data_manifest=f"{dirname(__file__)}/utils/data_info.yaml")

    def __init__(self):
        self.running_ruler = RunningRuler()
    
    def get_speed(self):
        times = []
        speeds = []
        car_types = []
        for times_track, centers_track, car_type_track in zip(self.data['times_records'],
                                                              self.data['centers_records'],
                                                              self.data['car_type_records']):
            # calculate speeds for a track
            times4speed_track, speeds_track = self.running_ruler.measure_speed_sequence(centers_track, times_track)
            
            if len(times4speed_track) > 0:
                times.append(times4speed_track[0] + (times4speed_track[-1] - times4speed_track[0]) / 2)
                speeds.append(sum(speeds_track) / len(speeds_track))
                car_types.append(car_type_track)

        times = np.array(times)
        speeds = np.array(speeds)
        car_types = np.array(car_types)

        df = pd.DataFrame({
            "time": times,
            "speed": speeds,
            "car_type": car_types,
        })

        df = df.set_index('time')
        df = df.sort_index()

        return df
