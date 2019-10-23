import importlib
import os
import pickle
from core.evaluation.environment import env_settings


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
    """

    def __init__(self, name: str, parameter_name: str, exp_name: str, run_id: int = None, checkpoint_id = None, flag: str = None):
        self.name = name
        self.exp_name = exp_name
        self.parameter_name = parameter_name
        self.run_id = run_id
        self.flag = flag
        self.checkpoint_id = checkpoint_id

        env = env_settings()

        tracker_module = importlib.import_module('core.tracker.{}'.format(self.name))

        self.parameters = self.get_parameters()
        self.tracker_class = tracker_module.get_tracker_class()

        self.default_visualization = getattr(self.parameters, 'visualization', False)
        self.default_debug = getattr(self.parameters, 'debug', 0)



    def get_parameters(self):
        """Get parameters."""

        param_module = importlib.import_module('core.parameter.{}.{}'.format(self.name, self.parameter_name))
        params = param_module.parameters(self.checkpoint_id)

        return params


