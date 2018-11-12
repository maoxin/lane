import cv2
import numpy as np
from sympy import Point, Line

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

    def get_foreground_mask(self, start_frame, end_frame, history=200):
        assert start_frame - history > 0, f"start should be more than history {history}"
        start_frame0 = start_frame - history

        ar_frame_ind = np.arange(start_frame0, end_frame+1, dtype='int')

        fgbg = cv2.createBackgroundSubtractorMOG2()

        fgmasks = []
        for i, frame_ind in enumerate(ar_frame_ind):
            frame, _ = self.__getitem__(frame_ind)
        
            fgmask = fgbg.apply(frame)
            if i >= 200:
                fgmasks.append(fgmask)

        return np.array(fgmasks, dtype='uint8')

    def __len__(self):
        return self.__total_frames
    
    def __getitem__(self, key):
        return self.__read_frame(frame_num=key)


class SingleObjectFGMaskImage(object):
    """
    in an front ground mask with just a single object, compute the hull of that
        object.
    """
    def __init__(self, fgmask, object_bbox):
        self.x0, self.y0, self.x1, self.y1 = object_bbox
        self.__x0_dilate = max(self.x0 - 40, 0)
        self.__x1_dilate = min(self.x1 + 40, fgmask.shape[1])
        self.__y0_dilate = max(self.y0 - 40, 0)
        self.__y1_dilate = min(self.y1 + 40, fgmask.shape[0])

        self.fgmask = fgmask
        self.fgmask_single_object = self.fgmask[self.__y0_dilate: self.__y1_dilate+1,
                                                self.__x0_dilate: self.__x1_dilate+1]

    def get_dilate_offset(self):
        return self.__x0_dilate, self.__y0_dilate

    def get_hull_of_object(self):
        ret, fgmask_single_object = cv2.threshold(self.fgmask_single_object, 127, 255, 0)

        kernel = np.ones((5, 5), np.uint8)
        fgmask_single_object = cv2.erode(fgmask_single_object, kernel, iterations=2)
        fgmask_single_object = cv2.dilate(fgmask_single_object, kernel, iterations=12)
        fgmask_single_object = cv2.erode(fgmask_single_object, kernel, iterations=10)
        # clear noise and fill hole

        _, contours, hierarchy = cv2.findContours(fgmask_single_object,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        external_contour_inds = [x[1] for x in np.where(hierarchy[: ,: ,3] == -1)]
        external_contours = [contours[i] for i in external_contour_inds]

        object_hull = None
        max_hull_area = 0
        for contour in external_contours:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)

            if hull_area > max_hull_area:
                max_hull_area = hull_area
                object_hull = hull

        return object_hull.reshape(-1, 2)

class SingleBusFGMaskImage(SingleObjectFGMaskImage):
    def __init__(self, fgmask, object_bbox):
        super().__init__(fgmask, object_bbox)

    def __rotate_point_anti_clockwise(self, ar_xy, angle_deg):
        angle_rad = np.deg2rad(angle_deg)

        rotate_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ])

        new_ar_xy = (rotate_matrix @ ar_xy.T).T

        return new_ar_xy
    
    def get_key_geometry(self, r_aclockwise=10):
        bus_hull = self.get_hull_of_object()

        bus_hull_rc = self.__rotate_point_anti_clockwise(bus_hull, -r_aclockwise)
        bus_hull_rac = self.__rotate_point_anti_clockwise(bus_hull, r_aclockwise)

        p1_ind = np.argmax(bus_hull_rc[:, 1])
        p2_ind = np.argmin(bus_hull_rac[:, 0])

        flow_theta = np.rad2deg(np.arctan2(*((bus_hull[p1_ind] - bus_hull[p2_ind])[::-1]))) * 0.8
        bus_hull_rc2 = self.__rotate_point_anti_clockwise(bus_hull, -flow_theta)

        p3_ind = np.argmax(bus_hull_rc2[:, 0])

        x0_dilate, y0_dilate = self.get_dilate_offset()
        p1 = bus_hull[p1_ind] + np.array([x0_dilate, y0_dilate])
        p2 = bus_hull[p2_ind] + np.array([x0_dilate, y0_dilate])
        p3 = bus_hull[p3_ind] + np.array([x0_dilate, y0_dilate])

        key_geometry = {
            'front_point': p1,
            'back_point': p2,
            'vertical_slope': (p3[1] - p1[1]) / (p3[0] - p1[0])
        }

        return key_geometry

if __name__ == '__main__':
    v = Video("/Volumes/U9/lane/cctv_ftg6.mp4")
    fgmask = v.get_foreground_mask(112590, 112590, 200)[0]

    x0, y0, x1, y1 = 361 * 1920 / 1022, 166 * 1080 / 575, 684 * 1920 / 1022, 499 * 1080 / 575
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    sb = SingleBusFGMaskImage(fgmask, (x0, y0, x1, y1))