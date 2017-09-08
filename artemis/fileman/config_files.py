import os
import sys
from six.moves.configparser import NoSectionError, NoOptionError, ConfigParser

__author__ = 'peter'


_CONFIG_OBJECTS = {}


def get_config_value(config_filename, section, option, default_generator=None, write_default=False, read_method=None, use_cashed_config=True):
    """
    Get a setting from a configuration file.  If none exists, you can optionally create one.  An example config file is
    the ~/.theanorc file, which may contain:

        [global]                    # <-- section
        floatX = float32            # <-- option (value is 'float32')
        device = gpu0

        [lib]
        cnmem = 1

        [cuda]
        root=/usr/local/cuda/

    :param config_filename: A configuration filename (eg '.theanorc').  Here we enforce the convention that files start
        with '.', and are stored in the user-home directory.
    :param section: Section in the config file (as referred by squared brackets: see example above)
    :param option: The option of interest (see above example)
    :param default_generator: A function that generates the property if it is not there, or just the default value if
        default_generator is not callable
    :param write_default: Set to true if property was not found and you want to write the default value into the file.
    :param read_method: The method to read your setting.
        If left at None (default) it just returns the string.
        If 'eval' it parses the setting into a python object
        If it is a function, it passes the value through the function before returning it.
    :param use_cashed_config: If set, will not read the config file from the file system but use the previously read and stored config file.
        Since the cashed config file might have been modified since reading it from disk, the returned value might be different from the value
        stored on the file on disk. In case write_default is set to True, the default value will be written to disk either way if no value has been found.
        If set to False, the original value will be returned without modifying the hashed version.
    :return: The value of the property of interest.
    """
    config_path = get_config_path(config_filename)
    default_used = False

    if write_default:
        assert default_generator is not None, "If you set write_default true, you must provide a function that can generate the default."

    if not os.path.exists(config_path):
        assert default_generator is not None, 'No config file "%s" exists, and you do not have any default value.' % (config_path, )
        value = default_generator()
        default_used = True
    else:
        config = _get_config_object(config_path, use_cashed_config=use_cashed_config)
        try:
            value = config.get(section, option)
        except (NoSectionError, NoOptionError) as err:
            if default_generator is None:
                raise
            else:
                value = default_generator() if callable(default_generator) else default_generator
                default_used = True

    if default_used and write_default:
        config = _get_config_object(config_path)
        if not config.has_section(section):
            config.add_section(section)
        config.set(section, option, value)
        with open(config_path, 'w') as f:
            config.write(f)

    if read_method == 'eval':
        value = eval(value)
    elif callable(read_method):
        value = read_method(value)
    return value


def _get_config_object(config_path, use_cashed_config=True):
    '''
    Returns a ConfigParser for the config file at the given path. If no file exists, an empty config file is created.
    :param config_path:
    :param use_cashed_config: If set to True, will return the previously created ConfigParser file (if previously created).
        If set to False, will re-read the config file from disk. If a ConfigParser was previously created, it will not be replaced!
    :return:
    '''
    if config_path not in _CONFIG_OBJECTS or not use_cashed_config:
        config = ConfigParser()
        if not os.path.exists(config_path):
            with open(config_path,'w') as f:
                config.write(f)
        else:
            config.read(config_path)
        if use_cashed_config:
            _CONFIG_OBJECTS[config_path] = config
    else:
        config = _CONFIG_OBJECTS[config_path]
    return config


def get_config_sections(config_filename):
    config_path = get_config_path(config_filename)
    sections = _get_config_object(config_path).sections()
    return sections


def get_home_dir():
    # This function exists because of some weirdness when running remotely.
    home_dir = os.path.expanduser('~')
    if os.path.exists(home_dir):
        return home_dir
    elif sys.platform=='linux2':
        home_dir = os.path.join('/home', os.getlogin())
    elif sys.platform=='darwin':
        home_dir==os.path.join('/Users', os.getlogin())
    assert os.path.exists(home_dir), 'The home directory "{}" seems not to exist.  This is odd, to say the least.'.format(home_dir)
    return home_dir

def get_config_path(config_filename):
    assert config_filename.startswith('.'), "We enforce the convention that configuration files must start with '.'"
    config_path = os.path.join(get_home_dir(), config_filename)
    return config_path

def set_non_persistent_config_value(config_filename, section, option, value):
    config_path = get_config_path(config_filename)
    config = _get_config_object(config_path)
    if not config.has_section(section):
        config.add_section(section)
    return config.set(section=section, option=option,value=value)


if __name__ == '__main__':
    config_path = get_config_path('.artemisrc')
    print('Contents of {}:\n-------------\n'.format(config_path))
    with open(config_path) as f:
        print(f.read())
