import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2 as cv
import time
import math

def IoU(rect1, rect2):
    # overlap

    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[0] + rect1[2], rect1[1] + rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[0] + rect2[2], rect2[1] + rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2-x1) * (y2-y1)

    target_a = (tx2-tx1) * (ty2 - ty1)

    inter = ww * hh
    overlap = inter / (area + target_a - inter)

    return overlap

def draw(image, name, bbox, score, iou):
    image = image.copy()
    bbox = [float(x) for x in bbox]
    x1, y1, x2, y2 = map(lambda x: int(round(x)), bbox)
    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
    cv.putText(image, "score: {:.2f} iou: {:.2f}".format(float(score), float(iou)), (30, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    cv.imwrite(name, image)

def get_axis_aligned_bbox(region):
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2

    return cx, cy, w, h

def cxy_wh_2_rect(pos, sz):
    return [pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]]

class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params

    def initialize(self, raw_im, image, state):
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, raw_im, image, gt):
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def _read_image(self, image_file: str):
        raw_im = cv.imread(image_file)
        im = cv.cvtColor(raw_im.copy(), cv.COLOR_BGR2RGB)
        return raw_im, im

    def _read_rgbd_image(self, image_file: str):
        raw_im = cv.imread(image_file)
        # print (image_file)
        im = cv.cvtColor(raw_im.copy(), cv.COLOR_BGR2RGB)
        depth_file = image_file.replace('color', 'depth').replace('jpg', 'png')
        # import pdb;pdb.set_trace()
        depth_im = cv.imread(depth_file, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
        return raw_im, im, depth_im

    def track_sequence_rgbd(self, sequence):
        """Run tracker on a sequence."""
        raw_im, image, depth_im = self._read_rgbd_image(sequence.frames[0])

        print("longterm term tracking")

        def trans(bbox):
            return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

        times = []
        start_time = time.time()
        cx, cy, w, h = get_axis_aligned_bbox(np.array(sequence.init_state))
        gt_bbox = [cx, cy, w, h]
        self.initialize(raw_im, image, depth_im, gt_bbox, init_online=True, video_name=sequence.name)

        init_time = getattr(self, 'time', time.time() - start_time)
        times.append(init_time)

        if self.params.visualization:
            self.init_visualization()
            self.visualize(image, gt_bbox)

        tracked_bb = [gt_bbox]
        scores = [float(1)]
        cnt = 1

        for frame in sequence.frames[1:]:
            print(frame)
            raw_im, image, depth_im = self._read_rgbd_image(frame)

            start_time = time.time()
            cx, cy, w, h = get_axis_aligned_bbox(np.array(sequence.ground_truth_rect[cnt]))
            # import pdb; pdb.set_trace()
            state, score = self.track(raw_im, image, depth_im, [cx, cy, w, h])
            times.append(time.time() - start_time)

            tracked_bb.append(state)
            score = float(score)
            if score > 1:
                score = float(1)
            elif score < 0:
                score = float(0)
            scores.append(score)

            if self.params.visualization:
                self.visualize(image, state)

            cnt += 1

        if getattr(self, 'record', None) == None:
            self.record = None

        return tracked_bb, times, scores, self.record

    def track_sequence(self, sequence):
        """Run tracker on a sequence."""
        raw_im, image = self._read_image(sequence.frames[0])

        if getattr(self.params, 'short_term', False):
            print("short term tracking")
            times, tracked_bb, scores = [], [], []
            start_time = time.time()
            times.append(time.time() - start_time)
            start_frame = 0
            for f, image_file in enumerate(sequence.frames):
                start_time = time.time()
                raw_im, image = self._read_image(sequence.frames[f])
                if f == start_frame:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(sequence.ground_truth_rect[f]))
                    gt_bbox = [cx, cy, w, h]
                    if getattr(self.params, 'restore_online', False):
                        read_online = getattr(self.params, 'read_online', False)
                        flag_read = read_online and (f==0)
                        self.initialize(raw_im, image, gt_bbox, init_online=(f==0), read_libs=flag_read, video_name=sequence.name)
                    else:
                        self.initialize(raw_im, image, gt_bbox, init_online=True, read_libs=(f==0), video_name=sequence.name)
                    tracked_bb.append([1, 1, 1, 1])
                elif f > start_frame:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(sequence.ground_truth_rect[f]))
                    gt_bbox = [cx - w * 0.5, cy - h * 0.5, w, h]
                    state, score = self.track(raw_im, image, [0, 0, 0, 0])
                    iou = IoU(gt_bbox, state)
                    if iou <= 0:
                        tracked_bb.append([2, 2, 2, 2])
                        start_frame = f + 5
                    else:
                        tracked_bb.append(state)
                else:
                    tracked_bb.append([0, 0, 0, 0])
                scores.append(0)
                times.append(time.time() - start_time)

        elif getattr(self.params, 'long_term', False):
            print("longterm term tracking")
            def trans(bbox):
                return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

            times = []
            start_time = time.time()
            cx, cy, w, h = get_axis_aligned_bbox(np.array(sequence.init_state))
            gt_bbox = [cx, cy, w, h]
            self.initialize(raw_im, image, gt_bbox, init_online=True, video_name=sequence.name)

            init_time = getattr(self, 'time', time.time() - start_time)
            times.append(init_time)

            if self.params.visualization:
                self.init_visualization()
                self.visualize(image, gt_bbox)

            tracked_bb = [gt_bbox]
            scores = [float(1)]
            cnt = 1

            for frame in sequence.frames[1:]:
                print (frame)
                raw_im, image = self._read_image(frame)

                start_time = time.time()
                cx, cy, w, h = get_axis_aligned_bbox(np.array(sequence.ground_truth_rect[cnt]))
                # import pdb; pdb.set_trace()
                state, score = self.track(raw_im, image, [cx,cy,w,h])
                times.append(time.time() - start_time)

                tracked_bb.append(state)
                score = float(score)
                if score > 1: score = float(1)
                elif score < 0:score = float(0)
                scores.append(score)

                if self.params.visualization:
                    self.visualize(image, state)

                cnt += 1

        else:
            raise ValueError('ST or LT?')

        if getattr(self, 'record', None) == None:
            self.record = None

        return tracked_bb, times, scores, self.record

    def track_webcam(self):
        """Run tracker with webcam."""

        class UIControl:
            def __init__(self):
                self.mode = 'init'  # init, select, track
                self.target_tl = (-1, -1)
                self.target_br = (-1, -1)
                self.mode_switch = False

            def mouse_callback(self, event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN and self.mode == 'init':
                    self.target_tl = (x, y)
                    self.target_br = (x, y)
                    self.mode = 'select'
                    self.mode_switch = True
                elif event == cv.EVENT_MOUSEMOVE and self.mode == 'select':
                    self.target_br = (x, y)
                elif event == cv.EVENT_LBUTTONDOWN and self.mode == 'select':
                    self.target_br = (x, y)
                    self.mode = 'track'
                    self.mode_switch = True

            def get_tl(self):
                return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

            def get_br(self):
                return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

            def get_bb(self):
                tl = self.get_tl()
                br = self.get_br()

                bb = [tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]]
                return bb

        ui_control = UIControl()
        cap = cv.VideoCapture(0)
        display_name = 'Display: ' + self.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        cv.setMouseCallback(display_name, ui_control.mouse_callback)

        if hasattr(self, 'initialize_features'):
            self.initialize_features()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_disp = frame.copy()

            if ui_control.mode == 'track' and ui_control.mode_switch:
                ui_control.mode_switch = False
                init_state = ui_control.get_bb()
                self.initialize(frame, init_state)

            # Draw box
            if ui_control.mode == 'select':
                cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)
            elif ui_control.mode == 'track':
                state = self.track(frame)
                state = [int(s) for s in state]
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)

            # Put text
            font_color = (0, 0, 0)
            if ui_control.mode == 'init' or ui_control.mode == 'select':
                cv.putText(frame_disp, 'Select target', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            elif ui_control.mode == 'track':
                cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ui_control.mode = 'init'

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def init_visualization(self):
        # plt.ion()
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()

    def visualize(self, image, state):
        self.ax.cla()
        self.ax.imshow(image)
        rect = patches.Rectangle((state[0], state[1]), state[2], state[3], linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(rect)

        if hasattr(self, 'gt_state') and False:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g',
                                     facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        plt.draw()
        plt.pause(0.001)

        if self.pause_mode:
            plt.waitforbuttonpress()

