import ConfigParser
import os

__author__ = 'peter'


_CONFIG = None


def get_artemis_config():
    """
    :return: A ConfigParser object, containing the information in the ~/.artemisrc file.  Create a default file if none
        exists.
    """
    global _CONFIG
    if _CONFIG is None:
        config_path = os.path.join(os.path.expanduser('~'), '.artemisrc')
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                f.write('[plotting]\nbackend: matplotlib')
        config = ConfigParser.ConfigParser(defaults = {
            'update_period': '1',
            })
        config.read(os.path.join(os.path.expanduser('~'), '.artemisrc'))
        _CONFIG = config
    return _CONFIG
