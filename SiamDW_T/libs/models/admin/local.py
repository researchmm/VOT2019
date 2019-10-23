import os
import os.path as osp
class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = osp.join('saved_checkpoint')  # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'  # Directory for tensorboard files.
        self.gtot_dir = ''
