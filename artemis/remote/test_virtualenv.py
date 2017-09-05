import pytest
from artemis.plotting.matplotlib_backend import get_plotting_server_address
from artemis.remote.utils import get_local_ips
from artemis.remote.virtualenv import check_diff_local_remote_virtualenv, get_remote_installed_packages


ip_address = get_plotting_server_address()
is_local = ip_address in get_local_ips()

@pytest.mark.skipif(ip_address="" or is_local, reason ="No sense for local ip")
def test_check_diff_local_remote_virtualenv():

    # original_virtual_env_value = get_config_value(".artemisrc", ip_address, "python")
    # import ConfigParser
    # import os
    #
    # Config = ConfigParser.ConfigParser()
    # Config.read(os.path.expanduser("~/.artemisrc"))
    # Config.set(section=ip_address,option="python",value="~/virtualenvs/test_env/bin/python")
    # with open(os.path.expanduser("~/.artemisrc"), 'wb') as configfile:
    #     Config.write(configfile)

    check_diff_local_remote_virtualenv(ip_address, auto_install=False, auto_upgrade=False,ignore_warnings=True)

    # Config.set(section=ip_address,option="python",value=original_virtual_env_value)
    # with open(os.path.expanduser("~/.artemisrc"), 'wb') as configfile:
    #     Config.write(configfile)

@pytest.mark.skipif(ip_address="" or is_local, reason ="No sense for local ip")
def test_get_remote_installed_packages():
    packages = get_remote_installed_packages(ip_address)
    print(packages)


if __name__ == "__main__":
    test_get_remote_installed_packages()
    test_check_diff_local_remote_virtualenv()