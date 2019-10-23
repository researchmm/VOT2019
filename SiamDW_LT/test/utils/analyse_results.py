import os
import numpy as np
import os.path as osp

video_keys = dict()
video_keys['18LT'] = ['ballet', 'bicycle', 'bike1', 'bird1', 'car1', 'car3', 'car6', 'car8', 'car9', 'car16',
                  'carchase', 'cat1', 'cat2', 'dragon', 'following', 'freestyle', 'group1', 'group2', 'group3',
                  'liverRun', 'longboard', 'nissan', 'person2', 'person4', 'person5', 'person7', 'person14',
                  'person17', 'person19', 'person20', 'rollerman', 'skiing', 'tightrope', 'uav1', 'yamaha']
video_keys['19LT'] = ['ballet','bicycle','bike1','bird1','boat','bull',
     'car1','car3','car6','car8','car9','car16','carchase','cat1',
     'cat2','deer','dog','dragon','f1','following','freesbiedog','freestyle',
     'group1','group2','group3','helicopter','horseride','kitesurfing',
     'liverRun','longboard','nissan','parachute','person2','person4',
     'person5','person7','person14','person17','person19','person20',
     'rollerman','sitcom','skiing','sup','tightrope','uav1','volkswagen',
     'warmup','wingsuit','yamaha']

pwd = osp.dirname(__file__)
result_dir = osp.join(pwd, "../tracking_results")
save_dir = osp.join(pwd, "../analyse_results")

def read_recordtxt(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.split(",")
            line[-1] = line[-1].split("\n")[0]
            line = [float(x) for x in line]
            data.append(line)
    return data

def split(record_data, thr, gt_thr):
    cnt = 0
    for idx, item in enumerate(record_data):
        if item[0] >= thr and item[1] >= gt_thr: # both pos
            cnt += 1
        elif item[0] <= thr and item[1] <= gt_thr: # both neg
            cnt += 1
    return cnt

def split_neg(record_data, thr, gt_thr):
    cnt, total = 0, 0
    for idx, item in enumerate(record_data):
        if item[1] <= gt_thr:
            total += 1
            if item[0] <= thr:
                cnt += 1
    return cnt, total

def split_pos(record_data, thr, gt_thr):
    cnt, total = 0, 0
    for idx, item in enumerate(record_data):
        if item[1] >= gt_thr:
            total += 1
            if item[0] >= thr:
                cnt += 1
    return cnt, total

def get_pre_rec(pos, neg, pos_gt, neg_gt):
    len_pre = len(pos)
    len_rec = len(pos_gt)
    inter = len(pos & pos_gt)
    precision, recall = 0, 0
    if len_pre > 0: precision = float(inter) / float(len_pre)
    if len_rec > 0: recall = float(inter) / float(len_rec)
    return precision, recall

def draw(arrx, arry, path, sub_name):
    import matplotlib.pyplot as plt
    from scipy.interpolate import spline
    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    T = arry
    power = arrx
    plt.gca().set_ylabel('precision')
    plt.gca().set_xlabel('recall')
    plt.scatter(T, power)
    for i in range(len(threshold)):
        plt.annotate(threshold[i], xy=(T[i], power[i]), xytext=(T[i] + 0.001, power[i] + 0.001))
    plt.savefig(os.path.join(path, "{}.png".format(sub_name)))
    plt.clf()

def calculate(record_data, gt_thr=0.7):
    precision = []
    total = len(record_data)
    threshold = np.arange(0.05, 1, 0.05).tolist()
    for thr in threshold:
        cnt = split(record_data, thr, gt_thr)
        pre = float(cnt) / float(total)
        precision.append(pre)

    return precision

def calculate_neg(record_data, gt_thr=0.7):
    precision = []
    threshold = np.arange(0.05, 1, 0.05).tolist()
    for thr in threshold:
        cnt, total = split_neg(record_data, thr, gt_thr)
        pre_neg = float(cnt) / float(total)
        cnt, total = split_pos(record_data, thr, gt_thr)
        pre_pos = float(cnt) / float(total)
        precision.append([pre_pos, pre_neg])

    return precision

def analyse_record(draw_pic=False):
    # (box_iou, gt_iou)
    tracker = "testiounet"
    results = ["res50_rpnbox_000"]
    dataset = '18LT'
    rpnbox_pos = False

    for result in results:
        type = result.split("_")[-2]
        save_pic_path = "/data/home/v-had/Tmsrasia_MSM/v-had/pre_rec/{}".format(result)
        if not os.path.exists(save_pic_path): os.mkdir(save_pic_path)

        record_dir = osp.join(result_dir, tracker, result)

        save_t_dir = osp.join(save_dir, tracker)
        if not osp.exists(save_t_dir): os.mkdir(save_t_dir)
        save_t_r_dir = osp.join(save_t_dir, result)
        if not osp.exists(save_t_r_dir): os.mkdir(save_t_r_dir)

        record_data = []

        for video in video_keys[dataset]:
            record_path = osp.join(record_dir, "{}_record.txt".format(video))
            record_datav = read_recordtxt(record_path)

            # pos_pre, pos_rec, neg_pre, neg_rec = calculate(record_datav, save_pic_path)
            # print("done")
            # while True: continue

            # print("For video: {}".format(video))
            # threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            # for thr, pos_pre, pos_rec, neg_pre, neg_rec in zip(threshold, pos_pre, pos_rec, neg_pre, neg_rec):
            #     print(" For thr: {}".format(thr))
            #     print("  pos_pre: {:.4f}, pos_rec: {:.4f}".format(pos_pre, pos_rec))
            #     print("  neg_pre: {:.4f}, neg_rec: {:.4f}".format(neg_pre, neg_rec))

            record_data += record_datav

        print("For result: {}".format(result))
        threshold = np.arange(0.05, 1, 0.05).tolist()
        if type == 'fixed01':
            precision = calculate(record_data, gt_thr=0.7) # []
        elif type == 'fixedgap':
            precision = calculate(record_data, gt_thr=0.5)
        elif type == 'rpnbox':
            gt = [0.1, 0.3, 0.5]
            if rpnbox_pos:
                precisions = [calculate(record_data, gt_thr=x) for x in gt]
            else:
                precisions = [calculate_neg(record_data, gt_thr=x) for x in gt]
        else:
            raise ValueError('Unsupported Type.')

        if type == 'rpnbox':
            for precision, gtt in zip(precisions, gt):
                precision = np.array(precision)
                print("*--gt_thr: {}".format(gtt))
                sum = 0
                for i in range(len(threshold)):
                    sum += precision[i]
                    print("*----threshold: {}, precision: {}".format(threshold[i], precision[i]))

                top5_mean = (precision[-1] + precision[-2] + precision[-3] + precision[-4] + precision[-5]
                             + precision[-6] + precision[-7] + precision[-8] + precision[-9] + precision[-10]) / 10.
                print("*----precision mean: {}".format(sum / len(threshold)))
                print("*---->=0.5 precision mean: {}".format(top5_mean))
        else:
            sum = 0
            for i in range(len(threshold)):
                sum += precision[i]
                print("*threshold: {}, precision: {}".format(threshold[i], precision[i]))

            top5_mean = (precision[-1] + precision[-2] + precision[-3] + precision[-4] + precision[-5]) / 5.
            print("*precision mean: {}".format(sum / 9.))
            print("*>=0.5 precision mean: {}".format(top5_mean))


if __name__ == '__main__':
    analyse_record()






