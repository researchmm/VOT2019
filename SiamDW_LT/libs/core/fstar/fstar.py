import cv2
import importlib

from libs.core.star import STAR
from libs.core.base import BaseTracker


class FSTAR(BaseTracker):

    def initialize(self, raw_im, image, state, *args, **kwargs):
        self.tracker_1 = STAR(self.params)

        tracker_flip, name_flip = getattr(self.params, 'flip_tracker', 'fstar'),\
                                  getattr(self.params, 'flip_name', 'res50_512_TLVC')
        self.flip_params = self.get_flip_para(tracker_flip, name_flip)

        self.tracker_2 = STAR(self.flip_params)

        self.tracker_1.initialize(raw_im, image, state, *args, **kwargs)

        raw_im = cv2.flip(raw_im, 1)
        image = cv2.flip(image, 1)

        self.imw2 = raw_im.shape[1] / 2.

        state[0] = (self.imw2 - state[0]) + self.imw2

        self.tracker_2.initialize(raw_im, image, state, *args, **kwargs)
        self.frame_num = 0

    def track(self, raw_im, image, gt):
        self.frame_num += 1
        raw_im_flip, image_flip = cv2.flip(raw_im.copy(), 1), cv2.flip(image.copy(), 1)
        box1, score1 = self.tracker_1.track(raw_im, image, gt)
        box2, score2 = self.tracker_2.track(raw_im_flip, image_flip, gt)

        bbox1 = [box1[0] + box1[2]/2., box1[1] + box1[3]/2., box1[2], box1[3]]
        bbox2 = [box2[0] + box2[2]/2., box2[1] + box2[3]/2., box2[2], box2[3]]

        # trans box back
        bbox2[0] = (self.imw2 - bbox2[0]) + self.imw2

        if self.IoU(bbox1, bbox2) < getattr(self.params, "flip_thr", 0.5):
            if score1 > score2:
                bbox = bbox1
                bbox[0] -= bbox[2] / 2.
                bbox[1] -= bbox[3] / 2.
                score = score1
            else:
                bbox = bbox2
                bbox[0] -= bbox[2] / 2.
                bbox[1] -= bbox[3] / 2.
                score = score2
        else:
            bbox = [(x + y)/2. for x, y in zip(bbox1, bbox2)]
            bbox[0] -= bbox[2] / 2.
            bbox[1] -= bbox[3] / 2.
            score = (score1 + score2) / 2.
        return bbox, score

    def IoU(self, rect1, rect2):
        # overlap
        import numpy as np
        x1, y1, x2, y2 = rect1[0] - rect1[2]/2., rect1[1] - rect1[3]/2., rect1[0] + rect1[2]/2., rect1[1] + rect1[3]/2.
        tx1, ty1, tx2, ty2 = rect2[0] - rect2[2]/2., rect2[1] - rect2[3]/2., rect2[0] + rect2[2]/2., rect2[1] + rect2[3]/2.
        xx1 = np.maximum(tx1, x1)
        yy1 = np.maximum(ty1, y1)
        xx2 = np.minimum(tx2, x2)
        yy2 = np.minimum(ty2, y2)
        ww = np.maximum(0, xx2 - xx1)
        hh = np.maximum(0, yy2 - yy1)
        area = (x2 - x1) * (y2 - y1)
        target_a = (tx2 - tx1) * (ty2 - ty1)
        inter = ww * hh
        overlap = inter / (area + target_a - inter)
        return overlap

    def get_flip_para(self, tracker, name):
        param_module = importlib.import_module('settings.{}.{}'.format(tracker, name))
        params = param_module.parameters()
        return params
