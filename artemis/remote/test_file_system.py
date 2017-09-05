import os

import pytest
import shutil

from artemis.config import get_artemis_config_value
from artemis.fileman.config_files import get_config_value
from artemis.fileman.local_dir import get_artemis_data_path
from artemis.plotting.matplotlib_backend import get_plotting_server_address
from artemis.remote.file_system import rsync, simple_rsync, check_config_file
from artemis.remote.utils import get_local_ips

ip_address = get_plotting_server_address()
is_local = ip_address in get_local_ips()


@pytest.mark.skipif(is_local, reason ="No sense for local ip")
def test_rsync():
    options = ["-r"]
    username = get_artemis_config_value(section=ip_address, option="username")

    from_path = get_artemis_data_path(relative_path="tmp/tests/", make_local_dir=True)
    with open(os.path.join(from_path, "test1"), "wb"):
        pass
    with open(os.path.join(from_path, "test2"), "wb"):
        pass

    to_path = "%s@%s:/home/%s/temp/"%(username, ip_address, username)
    assert rsync(options, from_path, to_path)
    shutil.rmtree(from_path)


@pytest.mark.skipif(is_local, reason ="No sense for local ip")
def test_simple_rsync():
    from_path = get_artemis_data_path(relative_path="tmp/tests/", make_local_dir=True)
    with open(os.path.join(from_path, "test1"), "wb"):
        pass
    with open(os.path.join(from_path, "test2"), "wb"):
        pass
    remote_path = "~/PycharmProjects/Distributed-VI/"

    assert simple_rsync(local_path=from_path, remote_path=remote_path, ip_address=ip_address, verbose=True)
    shutil.rmtree(from_path)

if __name__ == "__main__":
    test_simple_rsync()
    # test_rsync()
