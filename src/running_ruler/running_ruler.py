import numpy as np
from sympy import Point, Line
from video.undistortion import UnRadialDistortion
from video.video import SingleBusFGMaskImage

class Ruler(object):
    def __init__(self, length, p1, p2, slope1, slope2):
        self.length = length

        self.line1 = Line(Point(p1), slope=slope1)
        self.line2 = Line(Point(p2), slope=slope2)

        self.a1, self.b1, self.c1 = self.line1.coefficients
        self.a2, self.b2, self.c2 = self.line2.coefficients
        # a*x + b*y + c = 0
    
    def is_point_inside(self, point):
        """
        'point' could be single point as a tuple
            or a set of points as a 2-d array
        """

        if isinstance(point, np.ndarray):
            ret = ((self.a1 * point[:, 0] + self.b1 * point[:, 1] + self.c1) *
                   (self.a2 * point[:, 0] + self.b2 * point[:, 1] + self.c2))
        elif isinstance(point, tuple):
            x, y = point
            ret = (self.a1 * x + self.b1 * y + self.c1) * (self.a2 * x + self.b2 * y + self.c2) > 0
        else:
            raise Exception("point should be tuple or 2-d array")

        return ret

class RunningRuler(object):
    """
    running ruler is consisted of a sequence of multi Ruler objects

    for each ruler it will calculate the speed of a point travel through it
        and assign the speed in that ruler to the time the point leave the ruler
    """
    def __init__(self, p1s, p2s, slope1s, slope2s, lengths):
        self.rulers = []

        for p1, p2, slope1, slope2, length in zip(p1s, p2s, slope1s, slope2s, lengths):
            self.rulers.append(Ruler(length, p1, p2, slope1, slope2))

    def measure_speed_sequence(self, point_trace, time_trace):
        """
        point_trace = np.array([(x0, y0), (x1, y1), ...])
        time_trace = np.array([tm0, tm1, ...])
        """

        speeds = []
        times4record = []

        for ruler in self.rulers:
            is_point_inside_ruler = ruler.is_point_inside(point_trace)
            time_inside = time_trace[np.where(is_point_inside_ruler)]
            speed = ruler.length / (time_inside[-1] - time_inside[0])
            
            speeds.append(speed)
            times4record.append(time_inside[-1])

        speeds = [x for _, x in sorted(zip(times4record, speeds))]
        times4record = sorted(times4record)

        return times4record, speeds

class BuildRunningRuler(object):
    """
    build running rulers for speed measuring
    each hour has its own running ruler
    """

    def __init__(self, bus_records, video):
        """
        bus_records: the records of track of bboxs of buses.
        video: a video object
        """
        self.video = video
        pass

    def __call__(self):
        """
        return RunningRuler objects, for different time period
        """
        pass
    
    def __get_best_record(self, time_period):
        """
        undistort the bus records and find the straightest and smoothest
            one for construct running ruler in a time period
        
        the undistortion has to be made before to judge straightness
        but return the distorted and undistorted one both,
            the first is for front and back calculation, which is simpler with distorted one
            the second is for ruler building

        return the best record and the time trace for the best record
        """
        pass

    def __get_key_geometry(self, best_record, time_trace):
        """
        in the best records, from the bbox extract endpoints of the bottom line of the front side of
            each track of a bus

        the extraction combines records and video, the video should be background removed

        return the record of the track of the endpoints, and their vertical direction slope
        """
        
        start_frame = time_trace[0]
        end_frame = time_trace[1]

        fgmasks = self.video.get_foreground_mask(start_frame, end_frame)[time_trace-time_trace[0]]

        for bbox, fgmask, frame_ind in zip(best_record, fgmasks, time_trace):
            # image_bgr, _ = self.video[frame_ind]
            single_bus_fgmask_image = SingleBusFGMaskImage(fgmask, bbox)
            key_geometry = single_bus_fgmask_image.get_key_geometry()

        return key_geometry


    def __build_running_ruler(self, front_points, back_points,
                              front_vertical_slope, back_vertical_slope, length=11):
        """
        build running ruler based on the provided end point record
        """
        
        lengths = [length] * len(front_points)
        running_ruler = RunningRuler(front_points, back_points,
                                     front_vertical_slope, back_vertical_slope, lengths)
        
        return running_ruler