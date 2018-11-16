import numpy as np
from sympy import Point, Line
from video.undistortion import UnRadialDistortion
from video.video import SingleBusFGMaskImage
from video.video import Video
import json
import os
import matplotlib.pyplot as plt
import cv2

class Ruler(object):
    def __init__(self, length=None, p1=None, p2=None, slope1=None, slope2=None,
                 a1=None, b1=None, c1=None, a2=None, b2=None, c2=None):
        
        if not(p1 is None or p2 is None or slope1 is None or slope2 is None or length is None):
            self.length = length

            self.__line1 = Line(Point(p1), slope=slope1)
            self.__line2 = Line(Point(p2), slope=slope2)

            self.a1, self.b1, self.c1 = [float(x) for x in self.__line1.coefficients]
            self.a2, self.b2, self.c2 = [float(x) for x in self.__line2.coefficients]
            # a*x + b*y + c = 0
        elif not(length is None or a1 is None or b1 is None or c1 is None or a2 is None or b2 is None or c2 is None):
            self.a1, self.b1, self.c1, self.a2, self.b2, self.c2, self.length = a1, b1, c1, a2, b2, c2, length
        else:
            raise Exception("length/p1/p2/slope1/slope2 or a1/b1/c1/a2/b2/c2 should not be None")
    
    def is_point_inside(self, point):
        """
        'point' could be single point as a tuple
            or a set of points as a 2-d array
        """

        if isinstance(point, np.ndarray):
            ret = ((self.a1 * point[:, 0] + self.b1 * point[:, 1] + self.c1) *
                   (self.a2 * point[:, 0] + self.b2 * point[:, 1] + self.c2)) < 0
        elif isinstance(point, tuple):
            x, y = point
            ret = (self.a1 * x + self.b1 * y + self.c1) * (self.a2 * x + self.b2 * y + self.c2) < 0
        else:
            raise Exception("point should be tuple or 2-d array")

        return ret

    def __str__(self):
        ret = f"a1: {self.a1}, b1: {self.b1}, c1: {self.c1}\na2: {self.a2}, b2: {self.b2}, c2:{self.c2}"

        return ret

class RunningRuler(object):
    """
    running ruler is consisted of a sequence of multi Ruler objects

    for each ruler it will calculate the speed of a point travel through it
        and assign the speed in that ruler to the time the point leave the ruler
    """
    def __init__(self, p1s=None, p2s=None, slope1s=None, slope2s=None, lengths=None, use_default=True):
        self.rulers = []

        if use_default:
            with open(f'{os.path.dirname(__file__)}/running_ruler.json') as f:
                rulers_dict = json.load(f)
            
            for ruler in rulers_dict['rulers']:
                self.rulers.append(Ruler(a1=ruler['a1'], b1=ruler['b1'], c1=ruler['c1'],
                                         a2=ruler['a2'], b2=ruler['b2'], c2=ruler['c2'],
                                         length=ruler['length']))

        else:
            for p1, p2, slope1, slope2, length in zip(p1s, p2s, slope1s, slope2s, lengths):
                self.rulers.append(Ruler(length=length, p1=p1, p2=p2, slope1=slope1, slope2=slope2))

    def measure_speed_sequence(self, point_trace, time_trace):
        """
        point_trace = np.array([(x0, y0), (x1, y1), ...])
        time_trace = np.array([tm0, tm1, ...])

        can be used to measure speed less than length / min(time_interval)
            if length is 11m, min(time_interval) is 0.1 s, max speed is 110 m /s (396 km/h)
        """
        point_trace = np.array(point_trace)
        time_trace = np.array(time_trace)

        speeds = []
        times4record = []

        for ruler in self.rulers:
            is_point_inside_ruler = ruler.is_point_inside(point_trace)
            time_inside = time_trace[is_point_inside_ruler]

            if time_inside.size >= 2:
                if isinstance(time_inside[0], np.datetime64):
                    speed = ruler.length / (time_inside[-1] - time_inside[0]).item().total_seconds()
                else:
                    speed = ruler.length / (time_inside[-1] - time_inside[0])
                # m/s
                speed *= 3.6
                # km/h
                
                speeds.append(speed)
                times4record.append(time_inside[-1])

        speeds = [x for _, x in sorted(zip(times4record, speeds))]
        times4record = sorted(times4record)

        return times4record, speeds

    def save(self):
        rulers_dict = {'rulers': []}
        for ruler in self.rulers:
            rulers_dict['rulers'].append(
                {
                    'a1': ruler.a1,
                    'b1': ruler.b1,
                    'c1': ruler.c1,
                    'a2': ruler.a2,
                    'b2': ruler.b2,
                    'c2': ruler.c2,
                    'length': ruler.length,
                }
            )
        
        with open(f'{os.path.dirname(__file__)}/running_ruler.json', 'w') as f:
            json.dump(rulers_dict, f)
        
        return 0

class BuildRunningRuler(object):
    """
    build running rulers for speed measuring
    each hour has its own running ruler
    """

    def __init__(self, video, bus_records, bus_time_traces):
        """
        bus_records: the records of track of bboxs of buses.
            [[
                [
                    [x0, y0],
                    [bx0, by0],
                    [bx1, by1],
                ], ...
            ], ...]
        video: a video object
        """
        self.video = video
        self.bus_records = bus_records
        self.bus_time_traces = bus_time_traces

    def get_running_ruler(self):
        """
        return RunningRuler objects, for different time period
        """
        
        best_bus_records, best_bus_time_traces = self.__get_best_record()
        key_geometries = self.__get_key_geometry(best_bus_records, best_bus_time_traces)

        front_points = []
        back_points = []
        front_vertical_slope = []
        back_vertical_slope = []
        for key_geometry in key_geometries:
            front_points.append(key_geometry['front_point'])
            back_points.append(key_geometry['back_point'])
            front_vertical_slope.append(key_geometry['vertical_slope'])
            back_vertical_slope.append(key_geometry['vertical_slope'])

        running_ruler = self.__build_running_ruler(front_points, back_points,
                             front_vertical_slope, back_vertical_slope, length=11)

        return running_ruler
    
    def __get_best_record(self, score_thredhold=None, monitor_box=None):
        """
        undistort the bus records and find the straightest and smoothest
            ones for construct running ruler

        return the best records and the time trace for the best record,
            some bad box in a record should be removed, and the bus_time_trace should be updated
        """

        scores = []
        for bus_record in self.bus_records:
            straight_score = self.__get_straight_score(bus_record)
            smooth_score = self.__get_smooth_score(bus_record)
            speed_score = self.__get_speed_score(bus_record)
            
            scores.append(straight_score + smooth_score + speed_score)

        scores = np.array(scores)

        sorted_ind = np.argsort(scores)[::-1]
        scores = scores[sorted_ind]
        best_bus_records = self.bus_records[sorted_ind]
        best_bus_time_traces = self.bus_time_traces[sorted_ind]

        if not (score_thredhold is None):
            best_bus_records = best_bus_records[scores >= score_thredhold]
            best_bus_time_traces = best_bus_time_traces[scores >= score_thredhold]
        
        if not (monitor_box is None):
            x0, y0, x1, y1 = monitor_box

            for i in range(best_bus_records.shape[0]):
                ind2keep = (best_bus_records[i][:, 1, 0] >= x0) & \
                           (best_bus_records[i][:, 2, 0] <= x1) & \
                           (best_bus_records[i][:, 1, 1] >= y0) & \
                           (best_bus_records[i][:, 2, 1] <= y1)
                best_bus_records[i] = best_bus_records[i][ind2keep]
                best_bus_time_traces[i] = best_bus_time_traces[i][ind2keep]

        return best_bus_records, best_bus_time_traces

    def __get_straight_score(self, bus_record):
        return 0

    def __get_smooth_score(self, bus_record):
        # areas = abs((bus_record[:, 2] - bus_record[:, 1]).prob())
        # score = 1 / (areas.std() / areas.mean())
        
        return 0

    def __get_speed_score(self, bus_record):
        # score = 1 / len(bus_record)
        
        return 0

    def __get_key_geometry(self, best_records, time_traces):
        """
        in the best records, from the bbox extract endpoints of the bottom line of the front side of
            each track of a bus

        the extraction combines records and video, the video should be background removed

        return the record of the track of the endpoints, and their vertical direction slope
        """
        key_geometries = []
        for best_record, time_trace in zip(best_records, time_traces):
            start_frame = time_trace[0]
            end_frame = time_trace[-1]

            fgmasks = self.video.get_foreground_mask(start_frame, end_frame)[time_trace-time_trace[0]]

            for bbox, fgmask, frame_ind in zip(best_record, fgmasks, time_trace):
                # should pass the track angle for rotation later
                image_bgr, _ = self.video[frame_ind]
                bbox = bbox[1:].flatten()
                single_bus_fgmask_image = SingleBusFGMaskImage(image_bgr, fgmask, bbox)
                key_geometry = single_bus_fgmask_image.get_key_geometry()
                key_geometries.append(key_geometry)

        return key_geometries


    def __build_running_ruler(self, front_points, back_points,
                              front_vertical_slope, back_vertical_slope, length=11):
        """
        build running ruler based on the provided end point record
        """
        
        lengths = [length] * len(front_points)
        running_ruler = RunningRuler(front_points, back_points,
                                     front_vertical_slope, back_vertical_slope, lengths, use_default=False)
        
        return running_ruler
