import os

from artemis.remote.file_system import rsync, simple_rsync, check_config_file


def test_rsync():
    options = ["-r"]
    from_path = os.path.join(os.path.expanduser("~"), "temp/")
    to_path = "mreisser@146.50.28.6:/home/mreisser/temp/"
    assert rsync(options, from_path, to_path)


def test_simple_rsync():
    # from_path = "~/PycharmProjects/Distributed-VI/"
    # to_path = "~/temp/Distributed-VI/"
    local_path  = "~/PycharmProjects/Distributed-VI/"
    remote_path = "~/PycharmProjects/Distributed-VI/"
    address = "146.50.28.6"
    assert simple_rsync(local_path=local_path, remote_path=remote_path, ip_address=address, verbose=True)


if __name__ == "__main__":
    # test_simple_rsync()
    # test_rsync()
    ip_address = "146.50.28.6"
    check_config_file(ip_address)
