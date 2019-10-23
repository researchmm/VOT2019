import numpy as np
from settings.datasets.data import Sequence, BaseDataset, SequenceList


def VOTRGBD19DDataset():
    return VOTRGBD19DDatasetClass().get_sequence_list()

def get_video_keys():
    return VOTRGBD19DDatasetClass()._get_sequence_list()

class VOTRGBD19DDatasetClass(BaseDataset):
    """
    rgbd_rgbdataset dataset
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.rgbd_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        ext = 'png'
        start_frame = 1

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        frames = ['{base_path}/{sequence_path}/depth/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
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
        sequence_list = ['XMG_outside', 'backpack_blue', 'backpack_robotarm_lab_occ', 'backpack_room_noocc_1', 'bag_outside', 'bicycle2_outside', 'bicycle_outside', 'bottle_box', 'bottle_room_noocc_1', 'bottle_room_occ_1', 'box1_outside', 'box_darkroom_noocc_1', 'box_darkroom_noocc_10', 'box_darkroom_noocc_2', 'box_darkroom_noocc_3', 'box_darkroom_noocc_4', 'box_darkroom_noocc_5', 'box_darkroom_noocc_6', 'box_darkroom_noocc_7', 'box_darkroom_noocc_8', 'box_darkroom_noocc_9', 'box_humans_room_occ_1', 'box_room_noocc_1', 'box_room_noocc_2', 'box_room_noocc_3', 'box_room_noocc_4', 'box_room_noocc_5', 'box_room_noocc_6', 'box_room_noocc_7', 'box_room_noocc_8', 'box_room_noocc_9', 'box_room_occ_1', 'box_room_occ_2', 'boxes_backpack_room_occ_1', 'boxes_humans_room_occ_1', 'boxes_office_occ_1', 'boxes_room_occ_1', 'cart_room_occ_1', 'cartman', 'cartman_robotarm_lab_noocc', 'case', 'container_room_noocc_1', 'dog_outside', 'human_entry_occ_1', 'human_entry_occ_2', 'humans_corridor_occ_1', 'humans_corridor_occ_2_A', 'humans_corridor_occ_2_B', 'humans_longcorridor_staricase_occ_1', 'humans_shirts_room_occ_1_A', 'humans_shirts_room_occ_1_B', 'jug', 'mug_ankara', 'mug_gs', 'paperpunch', 'person_outside', 'robot_corridor_noocc_1', 'robot_corridor_occ_1', 'robot_human_corridor_noocc_1_A', 'robot_human_corridor_noocc_1_B', 'robot_human_corridor_noocc_2', 'robot_human_corridor_noocc_3_A', 'robot_human_corridor_noocc_3_B', 'robot_lab_occ', 'teapot', 'tennis_ball', 'thermos_office_noocc_1', 'thermos_office_occ_1', 'toy_office_noocc_1', 'toy_office_occ_1', 'trashcan_room_occ_1', 'trashcans_room_occ_1_A', 'trashcans_room_occ_1_B', 'trendNetBag_outside', 'trendNet_outside', 'trophy_outside', 'trophy_room_noocc_1', 'trophy_room_occ_1', 'two_mugs', 'two_tennis_balls']

        return sequence_list
