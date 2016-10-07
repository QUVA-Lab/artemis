from ConfigParser import NoSectionError, NoOptionError, ConfigParser
import os


__author__ = 'peter'


def get_config_value(config_filename, section, option, default_generator = None, write_default = False):
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
    :param default_generator: A function that generates the property if it is not there.
    :param write_default: Set to true if property was not found and you want to write the default value into the file.
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
        config = ConfigParser()
        config.read(config_path)
        try:
            value = config.get(section, option)
        except (NoSectionError, NoOptionError) as err:
            if default_generator is None:
                raise
            else:
                value = default_generator()
                default_used = True

    if default_used and write_default:
        config = ConfigParser()
        config.read(config_path)
        if not config.has_section(section):
            config.add_section(section)
        config.set(section, option, value)
        with open(config_path, 'w') as f:
            config.write(f)

    return value


def get_config_path(config_filename):
    assert config_filename.startswith('.'), "We enforce the convention that configuration files must start with '.'"
    config_path = os.path.join(os.path.expanduser('~'), config_filename)
    return config_path
