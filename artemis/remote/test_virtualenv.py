from artemis.fileman.config_files import get_config_value
from artemis.remote.virtualenv import check_diff_local_remote_virtualenv, get_remote_installed_packages


def test_check_diff_local_remote_virtualenv():
    ip_address = "146.50.28.6"
    original_virtual_env_value = get_config_value(".artemisrc", ip_address, "python")
    import ConfigParser
    import os

    Config = ConfigParser.ConfigParser()
    Config.read(os.path.expanduser("~/.artemisrc"))
    Config.set(section=ip_address,option="python",value="~/virtualenvs/test_env/bin/python")
    with open(os.path.expanduser("~/.artemisrc"), 'wb') as configfile:
        Config.write(configfile)

    check_diff_local_remote_virtualenv(ip_address, auto_install=True,ignore_warnings=True)

    Config.set(section=ip_address,option="python",value=original_virtual_env_value)
    with open(os.path.expanduser("~/.artemisrc"), 'wb') as configfile:
        Config.write(configfile)


def test_get_remote_installed_packages():
    ip_address = "146.50.28.6"
    packages = get_remote_installed_packages(ip_address)
    print packages



if __name__ == "__main__":
    test_get_remote_installed_packages()
    # check_diff_local_remote_virtualenv("146.50.28.6")
    # test_check_diff_local_remote_virtualenv()