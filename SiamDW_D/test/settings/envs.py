import os

class EnvSettings:
    def __init__(self):
        main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.results_path = '{}/tracking_results/'.format(main_path)
        self.otb_path = '{}/data/OTB2015'.format(main_path)
        self.nfs_path = ''
        self.uav_path = ''
        self.tpl_path = ''
        self.vot18_path = ''
        self.vot19_path = ''
        self.votlt18_path = '{}/data/VOT2018-LT'.format(main_path)
        self.votlt19_path = '{}/data/VOT2019-LT'.format(main_path)
        self.oxuva_path = '{}/data/long-term-tracking-benchmark/dataset/images'.format(main_path)
        self.oxuva_list = '{}/data/long-term-tracking-benchmark/dataset/tasks/test.csv'.format(main_path)
        self.rgbd_path = '{}/data/VOT2019-RGBD'.format(main_path)
        self.got10k_path = ''
        self.lasot_path = '{}/data/LaSOTBenchmark'.format(main_path)
        self.trackingnet_path = ''

