import importlib
import os
from settings.envs import EnvSettings

class Tracker:

    def __init__(self, name: str, parameter_name: str, exp_name: str, run_id: int = None):
        self.name = name
        self.parameter_name = parameter_name
        self.exp_name = exp_name
        self.run_id = run_id

        tracker_module = importlib.import_module('libs.core.{}'.format(self.name))

        self.parameters = self.get_parameters()
        self.tracker_class = tracker_module.get_tracker_class()

        env = EnvSettings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.exp_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.exp_name, self.run_id)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def run(self, seq, rgbd_flag=False):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
        """

        tracker = self.tracker_class(self.parameters)

        if not rgbd_flag:
            output_bb, execution_times, scores, record = tracker.track_sequence(seq)
        else:
            output_bb, execution_times, scores, record = tracker.track_sequence_rgbd(seq)

        self.parameters.free_memory()

        return output_bb, execution_times, scores, record

    def get_parameters(self):
        """Get parameters."""

        param_module = importlib.import_module('settings.{}.{}'.format(self.name, self.parameter_name))
        params = param_module.parameters()

        return params


