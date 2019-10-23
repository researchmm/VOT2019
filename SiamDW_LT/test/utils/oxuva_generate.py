import os
import cv2
import oxuva
import numpy as np

def rect_from_opencv(rect, imsize_hw):
    imheight, imwidth = imsize_hw
    xmin_abs, ymin_abs, width_abs, height_abs = rect
    xmax_abs = xmin_abs + width_abs
    ymax_abs = ymin_abs + height_abs
    return {
        'xmin': xmin_abs / imwidth,
        'ymin': ymin_abs / imheight,
        'xmax': xmax_abs / imwidth,
        'ymax': ymax_abs / imheight,
    }

def main():
    # runs = 5
    runs = 1
    score_thr = 0.3

    once_flag = True
    choice = 4

    tracker = "fstar"
    rst_name = "oxuva_000"
    base_dir = "/data/home/v-hongyuan/projects/lt_tracking/test/tracking_results_bak/"

    file_path = "/data/home/v-had/github/long-term-tracking-benchmark/dataset/tasks/test.csv"
    dataset_path = "/data/home/v-had/github/long-term-tracking-benchmark/dataset/images/test"

    save_base_dir = "./fstar_oxuva/"
    with open(file_path, "r") as f: tasks = oxuva.load_dataset_tasks_csv(f)

    for i in range(runs):
        if once_flag and i > 0: break
        if once_flag: i = choice
        print("Generating {}_0{:02d}".format(rst_name, i))
        # rst_dir = os.path.join(base_dir, tracker, "{}_0{:02d}".format(rst_name, i))
        rst_dir = os.path.join(base_dir, tracker, rst_name)
        # save_dir = os.path.join(save_base_dir, "{}_0{:02d}_thr_{}".format(rst_name, i, score_thr)).replace('.', '_')
        save_dir = save_base_dir
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        keys = tasks.keys()
        for key in keys:
            vid, obj = key
            b_file = os.path.join(rst_dir, "{}_{}.txt".format(vid, obj))
            s_file = os.path.join(rst_dir, "{}_{}_score.txt".format(vid, obj))
            bbox, scores = [], []
            with open(b_file, "r") as f:
                for line in f:
                    anno = line.split("\t")
                    anno[-1] = anno[-1].split("\n")[0]
                    anno = [float(x) for x in anno]
                    bbox.append(anno)
            with open(s_file, "r") as f:
                for idx, line in enumerate(f):
                    if idx == 0: continue
                    anno = line.split("\n")[0]
                    scores.append(float(anno))

            im_path = os.path.join(dataset_path, "{}".format(vid), "000000.jpeg")
            im = cv2.imread(im_path)
            im_height, im_width = im.shape[0], im.shape[1]

            preds = oxuva.SparseTimeSeries()
            for idx, score in enumerate(scores):
                rect = np.array(bbox[idx])
                rect = rect_from_opencv(rect, imsize_hw=(im_height, im_width))
                if score > score_thr:
                    preds[idx] = oxuva.make_prediction(present=True, score=score, **rect)
                else:
                    preds[idx] = oxuva.make_prediction(present=False, score=0)

            pred_file_path = os.path.join(save_dir, "{}_{}.csv".format(vid, obj))
            with open(pred_file_path, 'w') as fp:
                oxuva.dump_predictions_csv(vid, obj, preds, fp)

if __name__ == '__main__':
    main()
