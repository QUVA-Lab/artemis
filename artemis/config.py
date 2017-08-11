import os
from artemis.fileman.config_files import get_config_path, get_config_value

_DEFAULT_ARTEMIS_CONFIG = """
[plotting]
backend = matplotlib
update_period = 0.5
mode = safe
default_fig_size = (12, 8)

[experiments]
experiment_directory = ~/.artemis/experiments 
"""

_CONFIG_FILE_NAME = '.artemisrc'


def check_or_create_artemis_config():
    config_path = get_config_path(_CONFIG_FILE_NAME)
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            f.write(_DEFAULT_ARTEMIS_CONFIG)
    return _CONFIG_FILE_NAME


def get_artemis_config_value(section, option, default_generator = None, write_default = False, read_method=None):
    """
    Get a setting from the artemis configuration.
    See docstring for get_config_value
    """
    config_filename = check_or_create_artemis_config()
    return get_config_value(config_filename, section=section, option=option, default_generator=default_generator, write_default=write_default, read_method=read_method)
