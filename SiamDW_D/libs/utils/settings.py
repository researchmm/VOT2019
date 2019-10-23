import os


class EnvSettings:
    def __init__(self):
        main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        self.network_path = '{}/../test/networks/'.format(main_path)


def local_env_settings():
    settings = EnvSettings()
    return settings
