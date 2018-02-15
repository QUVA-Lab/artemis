import os
import pytest
import shutil

from artemis.config import get_artemis_config_value
from artemis.experiments.experiment_record import get_experiment_dir
from artemis.fileman.local_dir import get_artemis_data_path, get_local_dir, get_local_path
from artemis.plotting.matplotlib_backend import get_plotting_server_address
from artemis.remote.file_system import rsync, simple_rsync, mount_directory, unmount_directory, mount_experiment_directory
from artemis.remote.utils import get_local_ips

ip_address = get_plotting_server_address()
is_local = ip_address in get_local_ips()


@pytest.mark.skipif(is_local, reason="No sense for local ip")
def test_rsync():
    options = ["-r"]
    username = get_artemis_config_value(section=ip_address, option="username")

    from_path = get_artemis_data_path(relative_path="tmp/tests/", make_local_dir=True)
    with open(os.path.join(from_path, "test1"), "wb"):
        pass
    with open(os.path.join(from_path, "test2"), "wb"):
        pass

    to_path = "%s@%s:/home/%s/temp/" % (username, ip_address, username)
    assert rsync(options, from_path, to_path)
    shutil.rmtree(from_path)


@pytest.mark.skipif(is_local, reason="No sense for local ip")
def test_simple_rsync():
    from_path = get_artemis_data_path(relative_path="tmp/tests/", make_local_dir=True)
    with open(os.path.join(from_path, "test1"), "wb"):
        pass
    with open(os.path.join(from_path, "test2"), "wb"):
        pass
    remote_path = "~/PycharmProjects/Distributed-VI/"

    assert simple_rsync(local_path=from_path, remote_path=remote_path, ip_address=ip_address, verbose=True)
    shutil.rmtree(from_path)


@pytest.mark.skipif(True, reason="Requires locally configured network")
def test_mounting_directory():
    user = "mreisser"
    ip = "146.50.28.6"
    local_dir = get_local_dir("/home/mreisser/test_dir", True)
    remote_dir = "/home/mreisser/.artemis/experiments"
    options = ["cache=yes", "kernel_cache", "compression=no", "large_read", "Ciphers=arcfour",
               "idmap=file", "gidfile=/home/mreisser/.ssh/gidfile", "uidfile=/home/mreisser/.ssh/uidfile",
               # "IdentityFile=/home/mreisser/.ssh/id_rsa"
               ]
    assert os.listdir(local_dir) == []
    mount_directory(user=user, ip=ip, local_dir=local_dir, remote_dir=remote_dir, raise_exception=True, options=options)

    assert os.listdir(local_dir) != []
    with open(get_local_path(local_dir + "tmpfile"), "wb"):
        pass

    unmount_directory(local_dir=local_dir, raise_exception=True)
    assert os.listdir(local_dir) == []

@pytest.mark.skipif(True, reason="Requires locally configured network")
def test_mounting_mounted_directory():
    user = "mreisser"
    ip = "146.50.28.6"
    local_dir = get_local_dir("/Users/matthias/test_dir", True)
    remote_dir = "/home/mreisser/.artemis/experiments"
    options = ["cache=yes", "kernel_cache", "compression=no"]
    assert os.listdir(local_dir) == []
    mount_directory(user=user, ip=ip, local_dir=local_dir, remote_dir=remote_dir, raise_exception=True, options=options)
    mount_directory(user=user, ip=ip, local_dir=local_dir, remote_dir=remote_dir, raise_exception=True, options=options)

    assert os.listdir(local_dir) != []
    with open(get_local_path(local_dir + "tmpfile"), "wb"):
        pass
    unmount_directory(local_dir=local_dir, raise_exception=True)
    assert os.listdir(local_dir) == []

@pytest.mark.skipif(True, reason="Requires locally configured network")
def test_mounting_expeiriment_directory():
    user = "mreisser"
    ip = "146.50.28.6"
    local_dir = get_local_dir("/Users/matthias/test_dir", True)
    remote_dir = "/home/mreisser/.artemis/experiments"
    options = ["cache=yes", "kernel_cache", "compression=no"]
    mount_experiment_directory(user, ip, remote_dir=remote_dir, local_dir=local_dir, options=options, raise_exception=True)
    assert get_experiment_dir() == local_dir


if __name__ == "__main__":
    test_mounting_mounted_directory()
    # test_mounting_expeiriment_directory()
    # test_mounting_directory()
    # test_simple_rsync()
    # test_rsync()
