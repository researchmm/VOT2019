#!/usr/bin/python
import os.path as osp
import sys
import os
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = osp.dirname(osp.abspath(__file__))
lib_path = osp.join(this_dir, '..')
add_path(osp.abspath(lib_path))

# print (sys.path)
import cv2
import torch
import numpy as np
import argparse
import multiprocessing
from settings.exp import EXP
from libs.core.tracker_test import Tracker

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=list, default="0456", help='gpu id')
parser.add_argument('--num_workers', default=4, type=int, help='multi processing')
args = parser.parse_args()

exp_obj = EXP()
exp_pairs = exp_obj.pairs

def write_record(record, path):
    with open(path, "w") as f:
        for line in record:
            f.write(str(line[0]))
            f.write(",")
            f.write(str(line[1]))
            f.write("\n")

# def run_sequence(seq: Sequence, tracker: Tracker):
def run_sequence(gpu_id, idx_quene, result_queue):
    """Runs a tracker on a sequence."""

    # set gpu id
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    # torch.cuda.set_device(int(gpu_id))

    while True:
        success_flag = True
        exp_idx = idx_quene.get()
        tracker, seq = exp_pairs[exp_idx]
        base_results_path = '{}/{}'.format(tracker.results_dir, seq.name)
        results_path = '{}.txt'.format(base_results_path)
        scores_path = '{}_score.txt'.format(base_results_path)
        times_path = '{}_time.txt'.format(base_results_path)
        record_path = '{}_record.txt'.format(base_results_path)
        if os.path.exists(results_path):
            success_flag = False
            result_queue.put('idx: {} seq: {} Finished'.format(exp_idx, seq.name))


        print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

        if success_flag:
            try:
                tracked_bb, exec_times, scores, record = tracker.run(seq)
                success_flag = True
            except Exception as e:
                print(e)
                success_flag = False
                with open(results_path, "w") as f: f.write("not finishd")
                result_queue.put('idx: {} seq: {} Failed!'.format(exp_idx, seq.name))

        if success_flag:
            tracked_bb = np.array(tracked_bb).astype(float)
            exec_times = np.array(exec_times).astype(float)
            scores = np.array(scores).astype(float)

            print('FPS: {}'.format(len(exec_times) / exec_times.sum()))

            np.savetxt(results_path, tracked_bb, delimiter='\t', fmt='%d')
            np.savetxt(scores_path, scores, delimiter='\t', fmt='%f')
            np.savetxt(times_path, exec_times, delimiter='\t', fmt='%f')
            if record != None:
                write_record(record, record_path)

            result_queue.put('idx: {} seq: {} Success!'.format(exp_idx, seq.name))

def one_test(gpu_id, seq_idx):
    """Runs a tracker on a sequence."""

    # set gpu id
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    # torch.cuda.set_device(int(gpu_id))
    # import pdb; pdb.set_trace()

    success_flag = True
    tracker, seq = exp_pairs[seq_idx]
    base_results_path = '{}/{}'.format(tracker.results_dir, seq.name)
    results_path = '{}.txt'.format(base_results_path)
    scores_path = '{}_score.txt'.format(base_results_path)
    times_path = '{}_time.txt'.format(base_results_path)
    record_path = '{}_record.txt'.format(base_results_path)
    if os.path.exists(results_path):
        success_flag = False
        return success_flag


    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if success_flag:
        tracked_bb, exec_times, scores, record = tracker.run(seq, rgbd_flag=True)
        success_flag = True

    if success_flag:
        tracked_bb = np.array(tracked_bb).astype(float)
        exec_times = np.array(exec_times).astype(float)
        scores = np.array(scores).astype(float)

        print('FPS: {}'.format(len(exec_times) / exec_times.sum()))

        np.savetxt(results_path, tracked_bb, delimiter='\t', fmt='%d')
        np.savetxt(scores_path, scores, delimiter='\t', fmt='%f')
        np.savetxt(times_path, exec_times, delimiter='\t', fmt='%f')
        if record != None:
            write_record(record, record_path)

        return success_flag

if __name__ == '__main__':
    # seq_idx = 138
    seq_idx = 1
    success_flag = one_test('1', seq_idx)
    tracker, seq = exp_pairs[seq_idx]
    print ("For exp_paris: {}  {}/{} state is: {}".format(seq.name, seq_idx, len(exp_pairs), success_flag))

'''
if __name__ == '__main__':
    gpus = args.gpus
    ctx = multiprocessing.get_context('spawn')
    idx_queue = ctx.Queue()
    result_queue = ctx.Queue()

    # run_sequence_test('5', exp_pairs, 1)

    num_workers = args.num_workers
    workers = [
        ctx.Process(
            target=run_sequence,
            args=(gpus[i % len(gpus)],
                  idx_queue, result_queue))
        for i in range(num_workers)
    ]

    for w in workers:
        w.daemon = True
        w.start()

    for i in range(len(exp_pairs)):
        idx_queue.put(i)

    for i in range(len(exp_pairs)):
        output_log = result_queue.get()
        tracker, seq = exp_pairs[i]
        print (output_log)
        # print ("For exp_paris: {}  {}/{} state is: {}".format(seq.name, i, len(exp_pairs), success_flag))
'''
