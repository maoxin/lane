import os
import numpy as np
import json

from running_ruler.running_ruler import BuildRunningRuler, RunningRuler
from video.video import Video

def test_constructing_running_ruler():
    video_path = f"{os.path.dirname(__file__)}/../../data/cctv_ftg6.mp4"
    fp_objects = f"{os.path.dirname(__file__)}/../video/cctv_ftg6.mp4.json"

    video = Video(video_path=video_path)
    
    with open(fp_objects) as f:
        track = json.load(f)
    
    bus_records = [[]]
    bus_time_traces = [[]]

    for key in track['frames']:
        x0 = int(track['frames'][key][0]['x1'] * video.width / track['frames'][key][0]['width'])
        x1 = int(track['frames'][key][0]['x2'] * video.width / track['frames'][key][0]['width'])
        y0 = int(track['frames'][key][0]['y1'] * video.height / track['frames'][key][0]['height'])
        y1 = int(track['frames'][key][0]['y2'] * video.height / track['frames'][key][0]['height'])

        xc = int((x0 + x1) / 2)
        yc = int((y0 + y1) / 2)

        record = np.array([
            [xc, yc],
            [x0, y0],
            [x1, y1],
        ])
        bus_records[0].append(record)
        bus_time_traces[0].append( (int(key) - 1) * 30)

    bus_records = np.array(bus_records)
    bus_time_traces = np.array(bus_time_traces)

    running_ruler_builder = BuildRunningRuler(video, bus_records, bus_time_traces)
    running_ruler = running_ruler_builder.get_running_ruler()
    running_ruler.save()

    return 0

def test_measurement():
    fp_track_records = f"{os.path.dirname(__file__)}/track_sample.json"

    running_ruler = RunningRuler()
    with open(fp_track_records) as f:
        track_records = json.load(f)
    
    point_trace = track_records['points']
    time_trace = track_records['seconds_from_start']

    times4record, speeds = running_ruler.measure_speed_sequence(point_trace, time_trace)

    print("times:", times4record)
    print("speeds:", speeds)
