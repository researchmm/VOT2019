import importlib


class Tracker:

    def __init__(self, name: str, parameter_name: str):
        self.name = name
        self.parameter_name = parameter_name

        tracker_module = importlib.import_module('libs.core.{}'.format(self.name))

        self.parameters = self.get_parameters()
        self.tracker_class = tracker_module.get_tracker_class()

    def get_parameters(self):
        """Get parameters."""

        param_module = importlib.import_module('test.settings.{}.{}'.format(self.name, self.parameter_name))
        params = param_module.parameters()

        return params


