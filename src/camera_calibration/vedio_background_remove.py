import cv2
import numpy as np

class Video(object):
    def __init__(self, video_path='../data/video/MVI_7739.mp4'):
        self.__video = cv2.VideoCapture(video_path)
        self.__total_frames = int(self.__video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(self.__video.get(cv2.CAP_PROP_FPS))
        

    def __read_frame(self, frame_num):
        assert frame_num < self.__total_frames, f"Fail!\nFrame Num {frame_num} exceed threshold {self.__total_frames}\n"
        assert frame_num >= 0, f"Fail!\nFrame Num {frame_num} less than 0\n"

        self.__video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, image_bgr = self.__video.read()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        return image_bgr, image_rgb

    def __len__(self):
        return self.__total_frames
    
    def __getitem__(self, key):
        return self.__read_frame(frame_num=key)

def background_remove(video, start, end):
    history = 200
    assert start - history < 0, f"start should be more than history {history}"
    start0 = start - history

    ar_frame_ind = np.arange(start0, end+1, dtype='int')

    fgbg = cv2.createBackgroundSubtractorMOG2()

    fgmasks = []
    for i, frame_ind in enumerate(ar_frame_ind):
        frame = video[frame_ind]
    
        fgmask = fgbg.apply(frame)
        if i >= 200:
            fgmasks.append(fgmask)

    return np.array(fgmasks, dtype='uint8')

