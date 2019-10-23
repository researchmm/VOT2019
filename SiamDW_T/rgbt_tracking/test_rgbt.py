# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Clean testing scripts for SiamFC
# New: support GENE and TPE tuning
# ------------------------------------------------------------------------------

import _init_paths
# import matlab.engine
import os
import cv2
import random
import argparse
import numpy as np
from os.path import exists, join, dirname, realpath
from easydict import EasyDict as edict
from utils.utils import cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou

from core.evaluation.running import run, track_frame
from core.evaluation import Tracker

# eng = matlab.engine.start_matlab()  # for test eao in vot-toolkit

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def parse_args():
    """
    args for rgbt testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch RGBT Tracking Test')
    parser.add_argument('--dataset', default='VOT2019RGBT', help='dataset test')
    args = parser.parse_args()

    return args


class AASTracker(object):
    def __init__(self, colorimage, inimage, region):
        # init tracker
        self.tracker_rgb = Tracker('improved', 'unrestore_res50_RGB', 'RGBandT', 0, 14, flag='RGB')
        self.tracker_t = Tracker('improved', 'unrestore_res50_T', 'RGBandT', 0, 14, flag='T')

        self.tracker_rgb, self.tracker_t = run(self.tracker_rgb, self.tracker_t)  # re-define tracker

        # init tracker
        self.tracker_rgb = self.tracker_rgb.tracker_init(colorimage, region, flag='RGB')
        self.tracker_t = self.tracker_t.tracker_init(inimage, region, flag='T')

    def track(self, colorimage, inimage):
        self.tracker_rgb, self.tracker_t, state = track_frame(self.tracker_rgb, self.tracker_t, colorimage, inimage)

        return [state[0], state[1], state[2], state[3]]


def track(video, args):
    start_frame, toc = 0, 0

    # save result to evaluate
    tracker_path = os.path.join('result', args.dataset)

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in args.dataset:
        baseline_path = os.path.join(tracker_path, 'baseline')
        video_path = os.path.join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    else:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

    if os.path.exists(result_path):
        return  # for mult-gputesting

    regions = []
    lost = 0
    # for rgbt splited test

    in_image_files, rgb_image_files, gt = video['infrared_imgs'], video['visiable_imgs'], video['gt']


    for f, in_f in enumerate(in_image_files):

        in_im = cv2.imread(in_f)
        in_im = cv2.cvtColor(in_im, cv2.COLOR_BGR2RGB)   # align with training

        rgb_im = cv2.imread(rgb_image_files[f])
        rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)  # align with training

        tic = cv2.getTickCount()

        if f == start_frame:  # init
            print('===============> init tracker')
            tracker = AASTracker(rgb_im, in_im, gt[f])

            regions.append(1)
        elif f > start_frame:  # tracking
            state = tracker.track(rgb_im, in_im)

            pos = np.array([state[0], state[1]])
            sz = np.array([state[2], state[3]])

            location = cxy_wh_2_rect(pos, sz)
            b_overlap = poly_iou(gt[f], location) if 'VOT' in args.dataset else 1
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append(2)
                start_frame = f + 5
                lost += 1
        else:
            regions.append(0)

        toc += cv2.getTickCount() - tic

    with open(result_path, "w") as fin:
        if 'VOT' in args.dataset:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps  Lost {}'.format(video['name'], toc, f / toc, lost))


def main():
    args = parse_args()

    # prepare video
    dataset = load_dataset(args.dataset)
    video_keys = list(dataset.keys()).copy()

    # tracking all videos in benchmark
    for video in video_keys:
        track(dataset[video], args)



if __name__ == '__main__':
    main()

