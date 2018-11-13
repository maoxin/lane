import cv2
import numpy as np
from sympy import Point, Line
import matplotlib.pyplot as plt
import json
import os

class Video(object):
    def __init__(self, video_path='../data/video/MVI_7739.mp4'):
        self.__video = cv2.VideoCapture(video_path)
        self.__total_frames = int(self.__video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(self.__video.get(cv2.CAP_PROP_FPS))
        self.width = int(self.__video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.__video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        

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
    def __init__(self, image, fgmask, object_bbox, use_default=True):
        self.use_default = use_default

        self.x0, self.y0, self.x1, self.y1 = object_bbox
        # self.__x0_dilate = max(self.x0 - 40, 0)
        # self.__x1_dilate = min(self.x1 + 40, fgmask.shape[1])
        # self.__y0_dilate = max(self.y0 - 40, 0)
        # self.__y1_dilate = min(self.y1 + 40, fgmask.shape[0])
        self.__x0_dilate = self.x0
        self.__x1_dilate = self.x1
        self.__y0_dilate = self.y0
        self.__y1_dilate = self.y1

        self.fgmask = fgmask
        self.image = image
        self.fgmask_single_object = self.fgmask[self.__y0_dilate: self.__y1_dilate+1,
                                                self.__x0_dilate: self.__x1_dilate+1]
        self.image_single_object = self.image[self.__y0_dilate: self.__y1_dilate+1,
                                                self.__x0_dilate: self.__x1_dilate+1]

    def get_dilate_offset(self):
        return self.__x0_dilate, self.__y0_dilate

    def get_hull_of_object(self, resize_x=600, resize_y=600, k=None, op=None, cl=None):
        if self.use_default:
            try:
                with open('key_geometry_parameter.json') as f:
                    parameter = json.load(f)
                k = parameter['k']
                op = parameter['op']
                cl = parameter['cl']
            except FileNotFoundError:
                pass

        ret, fgmask_single_object = cv2.threshold(self.fgmask_single_object, 127, 255, 0)

        if (not (resize_x is None)) and (not (resize_y is None)):
            default_y_size, default_x_size = fgmask_single_object.shape
            fgmask_single_object = cv2.resize(fgmask_single_object, dsize=(resize_x, resize_y))

        kernel = np.ones((k, k), np.uint8)

        fgmask_single_object = cv2.morphologyEx(fgmask_single_object, cv2.MORPH_OPEN, kernel, iterations=op)
        fgmask_single_object = cv2.morphologyEx(fgmask_single_object, cv2.MORPH_CLOSE, kernel, iterations=cl)

        # clear noise and fill hole

        _, contours, hierarchy = cv2.findContours(fgmask_single_object,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        external_contour_inds = np.where(hierarchy[: ,: ,3].flatten() == -1)[0]
        external_contours = [contours[i] for i in external_contour_inds]

        object_hull = None
        max_hull_area = 0
        for contour in external_contours:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)

            if hull_area > max_hull_area:
                max_hull_area = hull_area
                object_hull = hull

        object_hull = object_hull.reshape(-1, 2)

        if (not (resize_x is None)) and (not (resize_y is None)):
            object_hull[:, 0] = object_hull[:, 0] * default_x_size / fgmask_single_object.shape[0]
            object_hull[:, 1] = object_hull[:, 1] * default_y_size / fgmask_single_object.shape[1]

            object_hull = object_hull.astype('int32')

        return object_hull

class SingleBusFGMaskImage(SingleObjectFGMaskImage):
    def __init__(self, image, fgmask, object_bbox, use_default=True):
        super().__init__(image, fgmask, object_bbox, use_default)

    def __rotate_point_anti_clockwise(self, ar_xy, angle_deg):
        angle_rad = np.deg2rad(angle_deg)

        rotate_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ])

        new_ar_xy = (rotate_matrix @ ar_xy.T).T

        return new_ar_xy
    
    def get_key_geometry(self, r_aclockwise=10, resize_x=600, resize_y=600, k=None, op=None, cl=None):
        bus_hull = self.get_hull_of_object(resize_x=resize_x, resize_y=resize_y, k=k, op=op, cl=cl)

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
            'width_point': p3,
            'vertical_slope': (p3[1] - p1[1]) / (p3[0] - p1[0])
        }

        return key_geometry

class ValidatorSingleBusFGMaskImage(object):
    def __init__(self, video_path="../../data/cctv_ftg6.mp4", fp_objects="cctv_ftg6.mp4.json",
                 fp_ground_truth='ground_truth.json'):
        with open(fp_objects) as f:
            self.records = json.load(f)
        with open(fp_ground_truth) as f:
            self.ground_truth = json.load(f)

        self.video = Video(video_path)
        self.frame_records = int(self.records['framerate'])
        self.frame_inds = sorted([int( (int(key) - 1) * self.video.frame_rate / self.frame_records) for key in self.records['frames']])
        self.start = self.frame_inds[0]
        self.end = self.frame_inds[-1]

        self.fgmasks = self.get_fg_frames()

        self.parameter = self.select_parameters()

    def get_parameter(self):
        return self.parameter


    def get_fg_frames(self):
        fgmasks = self.video.get_foreground_mask(self.start, self.end, 200)

        return fgmasks

    def select_parameters(self):
        os.makedirs("../../result/opencv_experience", exist_ok=True)

        max_score = 0
        selected_k = None
        selected_op = None
        selected_cl = None

        for k in range(2, 10):
            for op in range(1, 10):
                for cl in range(1, 10):
                    print(f"k: {k}, op: {op}, cl: {cl}")
                    score = 0
                    
                    plt.close()
                    ncols = 3
                    nrows = int(np.ceil(len(self.records) / 3))
                    fig, axes = plt.subplots(ncols=ncols, nrows=nrows)
                    axes = axes.flatten()

                    for i, key in enumerate(self.records['frames']):
                        record = self.records['frames'][key][0]
                        x0 = int(record['x1'] * self.video.width / record['width'])
                        x1 = int(record['x2'] * self.video.width / record['width'])
                        y0 = int(record['y1'] * self.video.height / record['height'])
                        y1 = int(record['y2'] * self.video.height / record['height'])

                        frame_ind = int( (int(key) - 1) * self.video.frame_rate / self.frame_records)
                        fgmask = self.fgmasks[frame_ind-self.start]
                        image = self.video[frame_ind][1]
                        sb = SingleBusFGMaskImage(image, fgmask, (x0, y0, x1, y1), use_default=False)
                        hull = sb.get_hull_of_object(resize_x=600, resize_y=600, k=k, op=op, cl=cl)
                        result = sb.get_key_geometry(resize_x=600, resize_y=600, k=k, op=op, cl=cl)

                        gt = self.ground_truth[key]
                        score += 1 / (
                            np.sqrt(sum( (np.array(result['front_point']) - np.array(gt['front_point'])) **2 )) +
                            np.sqrt(sum( (np.array(result['back_point'] ) - np.array(gt['back_point']))  **2 )) + 
                            np.sqrt(sum( (np.array(result['width_point']) - np.array(gt['width_point'])) **2 ))
                            + 0.01)
                        
                        img = cv2.drawContours(sb.image_single_object.copy(), [hull], -1, (0, 255, 0), 3)
                        cv2.circle(img, (result['front_point'][0] - x0, result['front_point'][1] - y0),
                                   6, (255, 0, 0), -1)
                        cv2.circle(img, (result['back_point'][0] - x0, result['back_point'][1] - y0),
                                   6, (255, 0, 0), -1)
                        cv2.circle(img, (result['width_point'][0] - x0, result['width_point'][1] - y0),
                                   6, (255, 0, 0), -1)
                        axes[i].imshow(img)
                        axes[i].set_title(f"frame: {key}")
                    
                    if score > max_score:
                        max_score = score
                        selected_k = k
                        selected_op = op
                        selected_cl = cl
                    
                    for ax in axes:
                        ax.axis('off')
                    plt.suptitle(f"kernel: {k}, op: {op}, cl: {cl}")
                    plt.savefig(f"../../result/opencv_experience/k{k}_o{op}_c{cl}.pdf")

        parameter = {
            "k": selected_k,
            "op": selected_op,
            "cl": selected_cl,
        }

        with open("key_geometry_parameter.json", 'w') as f:
            json.dump(parameter, f)
        
        return parameter


if __name__ == '__main__':
    validate = ValidatorSingleBusFGMaskImage()
