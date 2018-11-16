import os
import numpy as np
import yaml

class DataFileReader(object):
    """
    Receive multiple data file and merge them

    can be acess via ind (bigger ind for latter time), time, period, or car type

    the input information can be written in a json or yaml file
    """
    def __init__(self, fp_data_manifest=None,
                 fps=None, start_times=None, frame_rate=30, track_interval=0.1):
        if not(fp_data_manifest is None) and fps is None and start_times is None:
            with open(fp_data_manifest) as f:
                data_info = yaml.load(f.read())
            fps = data_info['fps']
            start_times = data_info['start_times']
        elif not (fps is None or start_times is None):
            pass
        else:
            raise Exception("data_info_path should be given, xor fps and start_times.")
        
        self.frame_rate = frame_rate
        self.track_interval = track_interval
        
        self.times_records = []
        self.start_time_records = []
        self.car_type_records = []
        self.centers_records = []
        self.bboxes_records = []
        for fp, start_time in zip(fps, start_times):
            start_time = np.datetime64(start_time)
            sub_times_records, sub_start_time_records, sub_car_type_records, \
            sub_centers_records, sub_bboxes_records = self.__read_from_file(fp, start_time)

            self.times_records += sub_times_records
            self.start_time_records += sub_start_time_records
            self.car_type_records += sub_car_type_records
            self.centers_records += sub_centers_records
            self.bboxes_records += sub_bboxes_records

        self.times_records = np.array(self.times_records)
        self.start_time_records = np.array(self.start_time_records)
        self.car_type_records = np.array(self.car_type_records)
        self.centers_records = np.array(self.centers_records)
        self.bboxes_records = np.array(self.bboxes_records)

    def __read_from_file(self, fp, start_time):
        with open(fp) as f:
            data = f.readlines()
        
        times_records = []
        start_time_records = []
        car_type_records = []
        centers_records = []
        bboxes_records = []
        for line in data:
            track = line.strip().split('; ')
            track_start_frame = int(track[0])
            track_start_time = start_time + np.timedelta64(track_start_frame // self.frame_rate, 's')
            track_times = track_start_time + np.arange(0, (len(track) - 2) * 100, 100, dtype='timedelta64[ms]')

            track_car_type = track[1]
            track_centers = np.array([np.array(eval(x)[0]) for x in track[2:]])
            track_bboxes = np.array([np.array(eval(x)[1]) for x in track[2:]])

            times_records.append(track_times)
            start_time_records.append(track_start_time)
            car_type_records.append(track_car_type)
            centers_records.append(track_centers)
            bboxes_records.append(track_bboxes)

        return times_records, start_time_records, car_type_records, centers_records, bboxes_records

    def get_data(self):
        ret = {
            "times_records": self.times_records,
            "start_time_records": self.start_time_records,
            "car_type_records": self.car_type_records,
            "centers_records": self.centers_records,
            "bboxes_records": self.bboxes_records,
        }

        return ret

    def __get__(self, obj, objtype):
        return self.get_data()

class TestDescriptor(object):
    data = DataFileReader(f"{os.path.dirname(__file__)}/data_info.yaml")

