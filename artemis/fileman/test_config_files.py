from six.moves.configparser import NoSectionError, NoOptionError
from pytest import raises
from artemis.fileman.config_files import get_config_path, get_config_value, set_non_persistent_config_value
import os
__author__ = 'peter'


def test_get_config_value():

    config_path = get_config_path('.testconfigrc')

    if os.path.exists(config_path):
        os.remove(config_path)

    value = get_config_value(config_filename='.testconfigrc', section='opts', option='setting1', default_generator=lambda: 'somevalue', write_default=True)
    assert value == 'somevalue'

    value = get_config_value(config_filename='.testconfigrc', section='opts', option='setting1', default_generator=lambda: 'someothervalue', write_default=True)
    assert value == 'somevalue'

    value = get_config_value(config_filename='.testconfigrc', section='opts', option='setting2', default_generator=lambda: 'blah', write_default=True)
    assert value == 'blah'

    value = get_config_value(config_filename='.testconfigrc', section='opts', option='setting1')
    assert value == 'somevalue'

    value = get_config_value(config_filename='.testconfigrc', section='opts', option='setting2')
    assert value == 'blah'

    with raises(NoSectionError):
        _ = get_config_value(config_filename='.testconfigrc', section='schmopts', option='setting3')

    with raises(NoOptionError):
        _ = get_config_value(config_filename='.testconfigrc', section='opts', option='setting3')

    with raises(AssertionError):
        _ = get_config_value(config_filename='.testconfigXXXrc', section='opts', option='setting3')

    set_non_persistent_config_value(config_filename='.testconfigrc', section='opts', option='setting2',value="bob")
    value = get_config_value(config_filename='.testconfigrc', section='opts', option='setting2')
    assert value == 'bob'

    value = get_config_value(config_filename='.testconfigrc', section='opts', option='setting2', use_cashed_config=False)
    assert value == 'blah'

    value = get_config_value(config_filename='.testconfigrc', section='opts', option='setting2')
    assert value == 'bob'

    set_non_persistent_config_value(config_filename='.testconfigrc', section='schmapts', option='setting2', value="bob")
    with raises(NoOptionError):
        _ = get_config_value(config_filename='.testconfigrc', section='schmapts', option='setting3')

    with raises(NoSectionError):
        _ = get_config_value(config_filename='.testconfigrc', section='schmapts', option='setting2', use_cashed_config=False)

    value = get_config_value(config_filename='.testconfigrc', section='schmapts', option='setting2')
    assert value == 'bob'

    os.remove(config_path)


if __name__ == '__main__':
    test_get_config_value()
