from itertools import product
from settings.datasets import VOTLT18Dataset, VOTLT19Dataset, OTBDataset, LaSOTDataset, VOTRGBD19RGBDataset
from libs.core.tracker_test import Tracker

class EXP:
    "Tracker and sequence pairs"
    def __init__(self):
        self.pairs = []
        # self.pairs = self.fstar_otb() + self.fstar_lasot() + self.fstar_rgbd()
        self.pairs = self.far_rgbd()

    def fstar_vltlt2018(self):
        trackers = [Tracker('fstar', 'res50_512_TLVC', 'votlt2018', i) for i in range(1)]
        dataset = VOTLT18Dataset()
        pair_list = [(tracker_info, seq) for tracker_info, seq in product(trackers, dataset)]
        return pair_list

    def fstar_vltlt2019(self):
        trackers = [Tracker('fstar', 'res50_512_TLVC', 'votlt2019', i) for i in range(1)]
        dataset = VOTLT19Dataset()
        pair_list = [(tracker_info, seq) for tracker_info, seq in product(trackers, dataset)]
        return pair_list

    def fstar_oxuva(self):
        trackers = [Tracker('fstar', 'res50_512_TLVC', 'oxuva', i) for i in range(1)]
        dataset = OXUVADataset()
        pair_list = [(tracker_info, seq) for tracker_info, seq in product(trackers, dataset)]
        return pair_list

    def fstar_otb(self):
        trackers = [Tracker('fstar', 'res50_512_TLVC', 'otb', i) for i in range(1)]
        dataset = OTBDataset()
        pair_list = [(tracker_info, seq) for tracker_info, seq in product(trackers, dataset)]
        return pair_list

    def fstar_lasot(self):
        trackers = [Tracker('fstar', 'res50_512_TLVC', 'lasot', i) for i in range(1)]
        dataset = LaSOTDataset()
        pair_list = [(tracker_info, seq) for tracker_info, seq in product(trackers, dataset)]
        return pair_list

    def fstar_rgbd(self):
        trackers = [Tracker('fstar', 'res50_512_TLVC', 'rgbd', i) for i in range(1)]
        dataset = VOTRGBD19RGBDataset()
        pair_list = [(tracker_info, seq) for tracker_info, seq in product(trackers, dataset)]
        return pair_list

    def far_rgbd(self):
        trackers = [Tracker('far_fusion', 'far_fusion', 'rgbd', i) for i in range(1)]
        dataset = VOTRGBD19RGBDataset()
        pair_list = [(tracker_info, seq) for tracker_info, seq in product(trackers, dataset)]
        return pair_list
