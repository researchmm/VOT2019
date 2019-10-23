import os
from core.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    abs_path = os.path.abspath(os.path.dirname(__file__))
    # Set your local paths here.

    settings.network_path = os.path.join(abs_path, '../../../snapshot')    # Where tracking networks are stored.
    settings.results_path = os.path.join(abs_path, '../../../tracking_results/')    # Where to store tracking results


    return settings

