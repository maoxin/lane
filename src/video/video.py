import cv2
import numpy as np
from sympy import Point, Line
import matplotlib.pyplot as plt
import json

plt.ion()

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
    def __init__(self, image, fgmask, object_bbox):
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

    def get_hull_of_object(self, resize_x=600, resize_y=600, kernel=2, op=1, cl=1):
        ret, fgmask_single_object = cv2.threshold(self.fgmask_single_object, 127, 255, 0)

        if (not (resize_x is None)) and (not (resize_y is None)):
            default_y_size, default_x_size = fgmask_single_object.shape
            fgmask_single_object = cv2.resize(fgmask_single_object, dsize=(resize_x, resize_y))

        kernel = np.ones((2, 2), np.uint8)

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
    def __init__(self, image, fgmask, object_bbox):
        super().__init__(image, fgmask, object_bbox)

    def __rotate_point_anti_clockwise(self, ar_xy, angle_deg):
        angle_rad = np.deg2rad(angle_deg)

        rotate_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ])

        new_ar_xy = (rotate_matrix @ ar_xy.T).T

        return new_ar_xy
    
    def get_key_geometry(self, r_aclockwise=10, resize_x=600, resize_y=600, kernel=2, op=1, cl=1):
        bus_hull = self.get_hull_of_object(resize_x=resize_x, resize_y=resize_y, kernel=kernel, op=op, cl=cl)

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

def get_fg_frames(frame_inds, start, end):
    fgmasks = v.get_foreground_mask(start, end, 200)

    return fgmasks, start, end

def select_parameters(records, fgmasks, start, end, frame_records, ground_truth):
    max_score = 0
    selected_kernel = None
    selected_op = None
    selected_cl = None

    for kernel in range(2, 10):
        for op in range(1, 10):
            for cl in range(1, 10):
                print(f"kernel: {kernel}, op: {op}, cl: {cl}")
                score = 0
                for i, key in enumerate(records['frames']):
                    record = records['frames'][key][0]
                    x0 = int(record['x1'] * v.width / record['width'])
                    x1 = int(record['x2'] * v.width / record['width'])
                    y0 = int(record['y1'] * v.height / record['height'])
                    y1 = int(record['y2'] * v.height / record['height'])

                    frame_ind = int( (int(key) - 1) * v.frame_rate / frame_records)
                    fgmask = fgmasks[frame_ind-start]
                    image = v[frame_ind][1]
                    sb = SingleBusFGMaskImage(image, fgmask, (x0, y0, x1, y1))
                    hull = sb.get_hull_of_object(resize_x=600, resize_y=600, kernel=kernel, op=op, cl=cl)
                    result = sb.get_key_geometry(resize_x=600, resize_y=600, kernel=kernel, op=op, cl=cl)

                    gt = ground_truth[key]
                    score += 1 / (
                        np.sqrt(sum( (np.array(result['front_point']) - np.array(gt['front_point'])) **2 )) +
                        np.sqrt(sum( (np.array(result['back_point'] ) - np.array(gt['back_point']))  **2 )) + 
                        np.sqrt(sum( (np.array(result['width_point']) - np.array(gt['width_point'])) **2 ))
                         + 0.01)
                
                if score > max_score:
                    max_score = score
                    selected_kernel = kernel
                    selected_op = op
                    selected_cl = cl

    parameter = {
        "kernel": selected_kernel,
        "op": selected_op,
        "cl": selected_cl,
    }

    with open("key_geometry_parameter.json", 'w') as f:
        json.dump(parameter, f)
    
    return parameter

def main(records, fgmasks, start, end, frame_records, resize_x=600, resize_y=600, kernel=2, op=1, cl=1):
    for i, key in enumerate(records['frames']):
        plt.close()
        record = records['frames'][key][0]
        x0 = int(record['x1'] * v.width / record['width'])
        x1 = int(record['x2'] * v.width / record['width'])
        y0 = int(record['y1'] * v.height / record['height'])
        y1 = int(record['y2'] * v.height / record['height'])

        frame_ind = int( (int(key) - 1) * v.frame_rate / frame_records)
        # if frame_ind == 112590:
        fgmask = fgmasks[frame_ind-start]
        image = v[frame_ind][1]
        sb = SingleBusFGMaskImage(image, fgmask, (x0, y0, x1, y1))
        hull = sb.get_hull_of_object(resize_x=resize_x, resize_y=resize_y, kernel=kernel, op=op, cl=cl)
        result = sb.get_key_geometry(resize_x=resize_x, resize_y=resize_y, kernel=kernel, op=op, cl=cl)

        img = cv2.drawContours(sb.image_single_object.copy(), [hull], -1, (0, 255, 0), 3)

        plt.imshow(image)
        input("")

        # print(result)
        # axes[0].imshow(sb.fgmask_single_object)
        # axes[i][0].imshow(fgmask)
        # axes[i][1].imshow(img)
        # axes[i // 3, i % 3].imshow(img)
        # axes[i // 3, i % 3].set_title(f"frame: {key}")

        # plt.imshow(v[112590][1][y0: y1+1, x0: x1+1])
    # plt.subplots_adjust(wspace=0, hspace=0)
    # for ax in axes.flatten():
        # ax.axis('off')
    # plt.suptitle(f'kernel: {kernel}, open: {op}, close: {cl}')

    # plt.savefig(f'../../result/opencv_experience/k{kernel}_o{op}_c{cl}.pdf')

if __name__ == '__main__':
    v = Video("../../data/cctv_ftg6.mp4")
    with open('cctv_ftg6.mp4.json') as f:
        records = json.load(f)

    frame_records = int(records['framerate'])
    frame_inds = sorted([int( (int(key) - 1) * v.frame_rate / frame_records) for key in records['frames']])
    start = frame_inds[0]
    end = frame_inds[-1]

    with open('ground_truth.json') as f:
        ground_truth = json.load(f)


    