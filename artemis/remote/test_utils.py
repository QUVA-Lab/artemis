import SocketServer

from artemis.config import get_artemis_config_value
from pytest import raises

from artemis.remote.utils import get_local_ips, check_ssh_connection, check_if_port_is_free, get_socket

ip_addresses = [get_artemis_config_value(section="tests", option="remote_server",default_generator=lambda: "127.0.0.1")]
# ip_addresses=["127.0.0.1"]
if ip_addresses[0] not in get_local_ips():
    ip_addresses.append("127.0.0.1")

def test_check_ssh_connections():
    for ip_address in ip_addresses:
        if ip_address not in get_local_ips():
            check_ssh_connection(ip_address)


def test_check_if_port_is_free():
    for ip_address in ip_addresses:
        if ip_address not in get_local_ips():
            with raises(AssertionError):
                check_if_port_is_free(ip_address,80)
        else:
            sock, port = get_socket(ip_address,7005)
            # port is now not free:
            with raises(SocketServer.socket.error):
                check_if_port_is_free(ip_address, port)
            sock.close()
            # now no error to be expected
            check_if_port_is_free(ip_address,port)

if __name__ == "__main__":
    test_check_ssh_connections()
    test_check_if_port_is_free()
