import cv2
import torch
import numpy as np
import importlib
from libs.core.base import BaseTracker
from libs.core.senet_far import SenetFar
from libs.core.resnext_far import ResnextFar

class FAR_FUSION(BaseTracker):

    def initialize_rgbd(self, raw_im, image, depth_im, state, *args, **kwargs):
        #### add tracker senet ####
        tracker_senet = 'senet_far'
        param_senet = 'senet'
        senet_params = self.get_tracker_para(tracker_senet, param_senet)
        self.tracker_senet = SenetFar(senet_params)
        self.tracker_senet.initialize_rgbd(raw_im, image, depth_im, state, *args, **kwargs)

        #### add tracker resnext ####
        tracker_resnext = 'resnext_far'
        param_resnext = 'resnext'
        resnext_params = self.get_tracker_para(tracker_resnext, param_resnext)
        self.tracker_resnext = ResnextFar(resnext_params)
        self.tracker_resnext.initialize_rgbd(raw_im, image, depth_im, state, *args, **kwargs)

        self.frame_num = 0

    def update_tracker(self, tracker, box):
        tracker.pos = torch.Tensor([box[1], box[0]])
        tracker.target_sz = torch.Tensor([box[3], box[2]])

    def concat_boxes(self, box1, score1, box2, score2):
        if self.IoU(box1, box2) < 0.5:
            if score1 > score2:
                bbox = box1
                score = score1
                self.update_tracker(self.tracker_senet, bbox)
            else:
                bbox = box2
                score = score2
                self.update_tracker(self.tracker_senet, bbox)
        else:
            bbox = [(x + y)/2. for x, y in zip(box1, box2)]
            score = (score1 + score2) / 2.
        return bbox, score

    def track_rgbd(self, raw_im, image, depth_im, gt_box):
        self.frame_num += 1
        # raw_im_flip, image_flip, depth_flip = cv2.flip(raw_im.copy(), 1), cv2.flip(image.copy(), 1), cv2.flip(depth_im.copy(), 1)
        # import pdb; pdb.set_trace()
        box_senet, score_senet = self.tracker_senet.track(raw_im, image, depth_im)
        box_resnext, score_resnext = self.tracker_resnext.track(raw_im, image, depth_im)

        bbox, score = self.concat_boxes(box_senet, score_senet, box_resnext, score_resnext)

        return bbox, score

    def IoU(self, rect1, rect2):
        # overlap
        x1, y1, x2, y2 = rect1[0], rect1[1], rect1[0] + rect1[2], rect1[1] + rect1[3]
        tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[0] + rect2[2], rect2[1] + rect2[3]
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

    def get_tracker_para(self, tracker, name):
        param_module = importlib.import_module('settings.{}.{}'.format(tracker, name))
        params = param_module.parameters()
        return params
