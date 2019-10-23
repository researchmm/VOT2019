import os
import math
import matplotlib.pyplot as plt

def split():
    path = "/data/home/v-had/github/VOT-2019/Hao/pytracking/pytracking/debug_ar.log"
    data = dict()
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v:"):
                words = line.split(" ")
                video = words[1]
                if data.get(video) == None:
                    data[video] = []
                data[video].append(line)

    save_dir = "/data/home/v-had/component"

    for item in data.keys():
        with open(os.path.join(save_dir, "{}.txt".format(item)), "w") as f:
            lst = data[item]
            for r in lst:
                f.write(r)

def draw_scatter(x, y, name):
    plt.plot(x, y, "ro", markersize=2)
    plt.savefig(name)
    plt.clf()

def draw_hist(x, name, ratio=-1):
    plt.hist(x, 150, histtype='bar', rwidth=0.8)
    plt.savefig(name)
    plt.clf()

def draw_analyse():
    video_keys = ['ballet', 'bicycle', 'bike1', 'bird1', 'boat', 'bull',
                          'car1',
                          'car3',
                          'car6',
                          'car8',
                          'car9',
                          'car16',
                          'carchase',
                          'cat1',
                          'cat2',
                          'deer',
                          'dog',
                          'dragon',
                          'f1',
                          'following',
                          'freesbiedog',
                          'freestyle',
                          'group1',
                          'group2',
                          'group3',
                          'helicopter',
                          'horseride',
                          'kitesurfing',
                          'liverRun',
                          'longboard',
                          'nissan',
                          'parachute',
                          'person2',
                          'person4',
                          'person5',
                          'person7',
                          'person14',
                          'person17',
                          'person19',
                          'person20',
                          'rollerman',
                          'sitcom',
                          'skiing',
                          'sup',
                          'tightrope',
                          'uav1',
                          'volkswagen',
                          'warmup',
                          'wingsuit',
                          'yamaha']
    base_path = '/data/home/v-had/component'
    save_dir = "/data/home/v-had/Tmsrasia_MSM/v-had/AR_component/"

    for video in video_keys:
        rx, ry, ax, ay = [], [], [], []
        r_abs, a_abs = [], []
        with open(os.path.join(base_path, "{}.txt".format(video)), "r") as f:
            for line in f:
                split_words = line.split(" ")
                if len(split_words) == 12:
                    if split_words[9] == 'return':
                        continue
                    if not math.isnan(float(split_words[9])):
                        ry.append(float(split_words[5]))
                        ay.append(float(split_words[7]))
                        rx.append(float(split_words[9]))
                        ax.append(float(split_words[11]))
                    else:
                        r_abs.append(float(split_words[5]))
                        a_abs.append(float(split_words[7]))

        video_path = os.path.join(save_dir, "{}".format(video))
        if not os.path.exists(video_path):
            os.mkdir(video_path)

        draw_scatter(rx, ry, os.path.join(video_path, "RPN_present.jpg"))
        draw_scatter(ax, ay, os.path.join(video_path, "ATOM_present.jpg"))
        draw_hist(r_abs, os.path.join(video_path, "RPN_absent.jpg"))
        draw_hist(a_abs, os.path.join(video_path, "ATOM_absent.jpg"))

def draw_case3():
    video_keys = ['ballet', 'bicycle', 'bike1', 'bird1', 'boat', 'bull',
                  'car1',
                  'car3',
                  'car6',
                  'car8',
                  'car9',
                  'car16',
                  'carchase',
                  'cat1',
                  'cat2',
                  'deer',
                  'dog',
                  'dragon',
                  'f1',
                  'following',
                  'freesbiedog',
                  'freestyle',
                  'group1',
                  'group2',
                  'group3',
                  'helicopter',
                  'horseride',
                  'kitesurfing',
                  'liverRun',
                  'longboard',
                  'nissan',
                  'parachute',
                  'person2',
                  'person4',
                  'person5',
                  'person7',
                  'person14',
                  'person17',
                  'person19',
                  'person20',
                  'rollerman',
                  'sitcom',
                  'skiing',
                  'sup',
                  'tightrope',
                  'uav1',
                  'volkswagen',
                  'warmup',
                  'wingsuit',
                  'yamaha']
    base_path = '/data/home/v-had/component'
    save_dir = "/data/home/v-had/Tmsrasia_MSM/v-had/AR_component/"

    for video in video_keys:
        aiou, riou = [], []
        acnt, rcnt, asum, rsum = 0, 0, 0, 0
        ais, ris = 0, 0
        with open(os.path.join(base_path, "{}.txt".format(video)), "r") as f:
            for line in f:
                if 'C using atom' in line:
                    s = line.split(" ")
                    iou = float(s[-1])
                    if math.isnan(iou):
                        acnt += 1
                    else:
                        aiou.append(iou)
                        ais += iou
                    asum += 1
                elif 'C using rpn' in line:
                    s = line.split(" ")
                    iou = float(s[-1])
                    if math.isnan(iou):
                        rcnt += 1
                    else:
                        riou.append(iou)
                        ris += iou
                    rsum += 1
        if asum > 0:
            print("for video: {} atom absent: {}".format(video, float(acnt) / float(asum)))
            print("atom iou avg: {}".format(ais / float(asum)))
        else:
            print("for video: {} no atom".format(video))
        if rsum > 0:
            print("for video: {} rpn absent: {}".format(video, float(rcnt) / float(rsum)))
            print("rpn iou avg: {}".format(ris / float(rsum)))
        else:
            print("for video: {} no rpn".format(video))

        video_path = os.path.join(save_dir, "{}".format(video))

        draw_hist(riou, os.path.join(video_path, "report_rpn_iou.jpg"))
        draw_hist(aiou, os.path.join(video_path, "report_atom_iou.jpg"))

def main():
    video_keys = ['ballet', 'bicycle', 'bike1', 'bird1', 'boat', 'bull',
                  'car1',
                  'car3',
                  'car6',
                  'car8',
                  'car9',
                  'car16',
                  'carchase',
                  'cat1',
                  'cat2',
                  'deer',
                  'dog',
                  'dragon',
                  'f1',
                  'following',
                  'freesbiedog',
                  'freestyle',
                  'group1',
                  'group2',
                  'group3',
                  'helicopter',
                  'horseride',
                  'kitesurfing',
                  'liverRun',
                  'longboard',
                  'nissan',
                  'parachute',
                  'person2',
                  'person4',
                  'person5',
                  'person7',
                  'person14',
                  'person17',
                  'person19',
                  'person20',
                  'rollerman',
                  'sitcom',
                  'skiing',
                  'sup',
                  'tightrope',
                  'uav1',
                  'volkswagen',
                  'warmup',
                  'wingsuit',
                  'yamaha']
    base_path = '/data/home/v-had/component'
    save_dir = "/data/home/v-had/Tmsrasia_MSM/v-had/AR_component/"

    for video in video_keys:
        num_abs, num_map = 0, 0
        cor_iou = []
        cor_cnt, cor_sum, cor_is = 0, 0, 0
        with open(os.path.join(base_path, "{}.txt".format(video)), "r") as f:
            for line in f:
                if 'rs' in line:
                    tmp = line.split(" ")
                    if len(tmp) == 12:
                        if math.isnan(float(tmp[-1])):
                            num_abs += 1
                if 'enter' in line:
                    tmp = line.split(" ")
                    if len(tmp) <= 6: continue
                    if math.isnan(float(tmp[-1])):
                        num_map += 1
                if 'cor' in line:
                    s = line.split(" ")
                    iou = float(s[-1])
                    if math.isnan(iou):
                        cor_cnt += 1
                    else:
                        cor_iou.append(iou)
                        cor_is += iou
                    cor_sum += 1
        if cor_sum > 0:
            print("for video: {} cor while absent ratio: {}".format(video, float(cor_cnt)/float(cor_sum)))
            if cor_sum > cor_cnt:
                print("avg iou: {}".format(cor_is / float(cor_sum - cor_cnt)))
        else:
            print("for video: {} no abs".format(video, float()))
        # if rsum > 0:
        #     print("for video: {} rpn map absent: {}".format(video, float(rcnt) / float(rsum)))
        #     print("rpn map iou avg: {}".format(ris / float(rsum)))
        # else:
        #     print("for video: {} no map return rpn".format(video))

        video_path = os.path.join(save_dir, "{}".format(video))

        draw_hist(cor_iou, os.path.join(video_path, "cor_iou.jpg"))


if __name__ == '__main__':
    main()