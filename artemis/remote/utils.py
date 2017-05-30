from __future__ import print_function
import SocketServer
import logging
import socket
import struct
import sys
from artemis.config import get_artemis_config_value

ARTEMIS_LOGGER = logging.getLogger('artemis')


def get_socket(address,port):
    '''
    Returns a socket, bound to the next free port starting from the given port.
    :param port:
    :return: Tuple (socket, used_port)
    '''
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            server_address = (address, port)
            sock.bind(server_address)
        except SocketServer.socket.error as exc:
            # ARTEMIS_LOGGER.info('Port', port, 'already in use')
            port += 1
            if exc.args[0] == 48 or exc.args[0] == 98:
                port += 1
            else:
                raise
        else:
            break
    return (sock,port)


def send_size(sock, data):
    try:
        sock.sendall(struct.pack('!I', len(data)))
        sock.sendall(data)
    except SocketServer.socket.error as exc:
        if exc.args[0] == 32:
            print("Broken pipe", file=sys.stderr)
            sys.exit(0)
        else:
            raise


def recv_bytes(sock, size):
    buf = b''
    while size:
        try:
            newbuf = sock.recv(size)
        except SocketServer.socket.error as exc:
            if exc.args[0] == 54:
                print("Connection reset by peer", file=sys.stderr)
                sys.exit(0)
            else:
                raise
        buf += newbuf
        size -= len(newbuf)
    return buf

def recv_size(sock):

    size_data = recv_bytes(sock, 4)
    size = struct.unpack('!I', size_data)[0]
    message = recv_bytes(sock,size)

    return message

def get_local_ips():
    '''
    Returns all the different ways that one can address the own machine.
    source: http://stackoverflow.com/a/274644/2068168
    :return: list of strings
    '''
    from netifaces import interfaces, ifaddresses, AF_INET

    ip_list = []
    for interface in interfaces():
        if AF_INET in ifaddresses(interface):
            for link in ifaddresses(interface)[AF_INET]:
                ip_list.append(link['addr'])
    return ip_list

def is_valid_ip(ip_address):
    '''
    checks if ip_address has a valid format
    :param address:
    :return:
    '''
    try:
        socket.inet_aton(ip_address)
        return True
    except:
        return False

def is_valid_port(port):
    '''
    checks if port is a valid string.
    This will make sure that any given string is numeric and between the range of 0 and 65535
    Source: http://stackoverflow.com/a/12968117/2068168
    :param port:
    :return:
    '''
    import re
    port_pattern = re.compile("^([0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])$")
    return port_pattern.match(port)


def get_remote_artemis_path(remote_ip):
    return get_artemis_config_value(
        section=remote_ip,
        option='artemis_path',
        default_generator=lambda: raw_input('Specify Remote Artemis Installation Path: '),
        write_default=True
        )
