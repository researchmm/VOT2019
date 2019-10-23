import numpy as np
from settings.datasets.data import Sequence, BaseDataset, SequenceList


def VOTLT18Dataset():
    return VOTLT18DatasetClass().get_sequence_list()

def get_video_keys():
    return VOTLT18DatasetClass()._get_sequence_list()

class VOTLT18DatasetClass(BaseDataset):
    """
    VOTLT2018 dataset
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.votlt18_path
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
        sequence_list= ['ballet',
                        'bicycle',
                        'bike1',
                        'bird1',
                        'car1',
                        'car3',
                        'car6',
                        'car8',
                        'car9',
                        'car16',
                        'carchase',
                        'cat1',
                        'cat2',
                        'dragon',
                        'following',
                        'freestyle',
                        'group1',
                        'group2',
                        'group3',
                        'liverRun',
                        'longboard',
                        'nissan',
                        'person2',
                        'person4',
                        'person5',
                        'person7',
                        'person14',
                        'person17',
                        'person19',
                        'person20',
                        'rollerman',
                        'skiing',
                        'tightrope',
                        'uav1',
                        'yamaha']

        return sequence_list
