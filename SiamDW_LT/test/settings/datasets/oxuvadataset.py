import numpy as np
import oxuva
import os
import cv2
from settings.datasets.data import Sequence, BaseDataset, SequenceList


def OXUVADataset():
    return OXUVADatasetClass().get_sequence_list()


class OXUVADatasetClass(BaseDataset):
    """oxuva dataset """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.oxuva_path

    def get_sequence_list(self):
        file_path = self.env_settings.oxuva_list
        with open(file_path, "r") as f: tasks = oxuva.load_dataset_tasks_csv(f)
        self.tasks = tasks
        return SequenceList([self._construct_sequence(s) for s in tasks.keys()])

    def _construct_sequence(self, sequence_info):
        nz = 6
        ext = 'jpeg'
        task = self.tasks[sequence_info]
        frames = ['{base_path}/test/{vid_name}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, vid_name=sequence_info[0],
                                                                           frame=frame_num, nz=nz, ext=ext)
                  for frame_num in range(task.init_time + 1, task.last_time + 1)]

        sequence_name = "{}_{}".format(sequence_info[0], sequence_info[1])

        shape = cv2.imread(frames[0]).shape
        imw, imh = shape[1], shape[0]

        rect = task.init_rect
        xmin, ymin, xmax, ymax = rect['xmin'] * imw, rect['ymin'] * imh, rect['xmax'] * imw, rect['ymax'] * imh
        w, h = xmax - xmin, ymax - ymin
        init_rect = np.array([[xmin, ymin, w, h]])
        gt_rect = np.repeat(init_rect, task.last_time + 1, axis=0)

        return Sequence(sequence_name, frames, gt_rect)

    def __len__(self):
        return len(self.sequence_list)


if __name__ == '__main__':
    dataset = OXUVADataset()
