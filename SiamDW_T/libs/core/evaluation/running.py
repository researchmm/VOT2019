import numpy as np
import multiprocessing
import os
import time
import torch
from itertools import product
from .utils import poly_iou
from core.evaluation import Sequence, Tracker

def write_record(record, path):
    with open(path, "w") as f:
        for line in record:
            f.write(str(line[0]))
            f.write(",")
            f.write(str(line[1]))
            f.write("\n")

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


def strategy1(bbox_rgb, score_rgb, bbox_t, score_t):
    """
    fusion according to score
    """
    score_rgb_new = score_rgb / (score_rgb + score_t)
    score_t_new = score_t / (score_rgb + score_t)
    bbox = float(score_rgb_new) * np.array(bbox_rgb) + float(score_t_new) * np.array(bbox_t)

    pos = np.array([bbox[1] + bbox[3]/2, bbox[0] + bbox[2]/2], dtype=np.float32)
    sz = np.array([bbox[3], bbox[2]], dtype=np.float32)

    return bbox, torch.from_numpy(pos), torch.from_numpy(sz)



def track_frame(tracker_rgb, tracker_t, im_rgb, im_t):
    """Run tracker on a frame."""
    # get rgb random bboxes
    iou_features_rgb, init_boxes_rgb = tracker_rgb.track_getrandom(im_rgb, im_rgb, [0, 0, 0, 0])
    iou_features_t, init_boxes_t = tracker_t.track_getrandom(im_t, im_t, [0, 0, 0, 0])
    init_boxes = torch.cat([init_boxes_rgb, init_boxes_t])


    output_boxes, output_iou = tracker_rgb.refine_target_box_final(iou_features_rgb, iou_features_t,init_boxes, tracker_t, gt=[0, 0, 0, 0])

    tracker_rgb.refine_target_box_merge(output_boxes, output_iou)
    tracker_t.refine_target_box_merge(output_boxes, output_iou)


    # rgb results and rgb score
    state_rgb, score_rgb = tracker_rgb.track_final()
    state_t, score_t = tracker_t.track_final()

    return tracker_rgb, tracker_t, state_t


def run(tracker_rgb, tracker_t):
    """Run tracker on sequence.
    args:
        seq: Sequence to run the tracker on.
        visualization: Set visualization flag (None means default value specified in the parameters).
        debug: Set debug level (None means default value specified in the parameters).
    """

    tracker_rgb = tracker_rgb.tracker_class(tracker_rgb.parameters)
    tracker_t = tracker_t.tracker_class(tracker_t.parameters)


    # output_bb, execution_times, scores_rgb, scores_t = track_sequence(seq, tracker_rgb, tracker_t)

    # free memory
    # tracker_rgb.params.free_memory()
    # tracker_t.params.free_memory()

    return tracker_rgb, tracker_t
