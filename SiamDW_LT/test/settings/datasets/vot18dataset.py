import numpy as np
from settings.datasets.data import Sequence, BaseDataset, SequenceList


def VOT18Dataset():
    return VOT18DatasetClass().get_sequence_list()

def get_video_keys():
    return VOT18DatasetClass()._get_sequence_list()

class VOT18DatasetClass(BaseDataset):
    """VOT2018 dataset

    Publication:
        The sixth Visual Object Tracking VOT2018 challenge results.
        Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pfugfelder, Luka Cehovin Zajc, Tomas Vojir,
        Goutam Bhat, Alan Lukezic et al.
        ECCV, 2018
        https://prints.vicos.si/publications/365

    Download the dataset from http://www.votchallenge.net/vot2018/dataset.html"""
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vot18_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        ext = 'jpg'
        start_frame = 1

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                  sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
                  for frame_num in range(start_frame, end_frame+1)]

        if ground_truth_rect.shape[1] == 4:
            ground_truth_rect = np.column_stack((ground_truth_rect[:, 0],
                                                 ground_truth_rect[:, 1],
                                                 ground_truth_rect[:, 0],
                                                 ground_truth_rect[:, 1] + ground_truth_rect[:, 3],
                                                 ground_truth_rect[:, 0] + ground_truth_rect[:, 2],
                                                 ground_truth_rect[:, 1] + ground_truth_rect[:, 3],
                                                 ground_truth_rect[:, 0] + ground_truth_rect[:, 2],
                                                 ground_truth_rect[:, 1]))
        return Sequence(sequence_name, frames, ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list= ['ants1',
                        'ants3',
                        'bag',
                        'ball1',
                        'ball2',
                        'basketball',
                        'birds1',
                        'blanket',
                        'bmx',
                        'bolt1',
                        'bolt2',
                        'book',
                        'butterfly',
                        'car1',
                        'conduction1',
                        'crabs1',
                        'crossing',
                        'dinosaur',
                        'drone_across',
                        'drone_flip',
                        'drone1',
                        'fernando',
                        'fish1',
                        'fish2',
                        'fish3',
                        'flamingo1',
                        'frisbee',
                        'girl',
                        'glove',
                        'godfather',
                        'graduate',
                        'gymnastics1',
                        'gymnastics2',
                        'gymnastics3',
                        'hand',
                        'handball1',
                        'handball2',
                        'helicopter',
                        'iceskater1',
                        'iceskater2',
                        'leaves',
                        'matrix',
                        'motocross1',
                        'motocross2',
                        'nature',
                        'pedestrian1',
                        'rabbit',
                        'racing',
                        'road',
                        'shaking',
                        'sheep',
                        'singer2',
                        'singer3',
                        'soccer1',
                        'soccer2',
                        'soldier',
                        'tiger',
                        'traffic',
                        'wiper',
                        'zebrafish1']

        return sequence_list
