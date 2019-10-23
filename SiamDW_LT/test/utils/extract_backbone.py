import os
import torch
import collections

def main():
    chk_path = '/data/home/v-had/github/VOT-2019/Hao/pytracking/pytracking/networks/res50_unfix2_TLVC.pth.tar'
    save_path = '/data/home/v-had/.torch/models/res50_unfix.pth'
    example = '/data/home/v-had/.torch/models/resnet50-19c8e357.pth'
    chk = torch.load(chk_path)
    model = chk['net']
    new_backbone = collections.OrderedDict()
    for key in model.keys():
        if 'feature_extractor' in key:
            new_key = key[18:]
            new_backbone[new_key] = model[key]
    torch.save(new_backbone, save_path)



if __name__ == '__main__':
    main()