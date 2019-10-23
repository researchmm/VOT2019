#!/usr/bin/python
import os.path as osp
import sys
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..')
add_path(lib_path)

import cv2
import numpy as np

from star_tracking import vot
from libs.core.tracker_matlab import Tracker

class STARTracker(object):
    def __init__(self, raw_im, image, region):
        # init tracker
        self.tracker = Tracker('fstar', 'res50_512_TLVC')
        self.tracker = self.tracker.tracker_class(self.tracker.parameters)
        cx, cy, w, h = self.get_axis_aligned_bbox(region)
        init_state = [float(cx), float(cy), float(w), float(h)]
        self.tracker.initialize(raw_im, image, init_state, init_online=True, video_name=None)

    def get_axis_aligned_bbox(self, region):
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
            cx = x + w / 2
            cy = y + h / 2

        return cx, cy, w, h

    def track(self, raw_im, image):
    
        state, score = self.tracker.track(raw_im, image)

        if score > 1: score = float(score)
        elif score < 0: score = float(score)
        else: score = float(score)

        return vot.Rectangle(state[0], state[1], state[2], state[3]), score


# handle = vot.VOT("rectangle")
handle = vot.VOT("polygon")
selection = handle.region()
gt_init = np.array(selection.points)
gt_init = np.reshape(gt_init, (8))


imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

raw_im = cv2.imread(imagefile, cv2.IMREAD_COLOR)
image = cv2.cvtColor(raw_im.copy(), cv2.COLOR_BGR2RGB)
tracker = STARTracker(raw_im, image, gt_init)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    raw_im = cv2.imread(imagefile, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(raw_im.copy(), cv2.COLOR_BGR2RGB)
    region, confidence = tracker.track(raw_im, image)
    handle.report(region, confidence)

