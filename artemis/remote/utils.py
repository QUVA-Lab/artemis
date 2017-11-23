from __future__ import print_function

import subprocess
from six.moves import socketserver
from six.moves import input
from six.moves import queue as Queue

import getpass
import logging
import socket
import struct
import sys
import os
import time

from artemis.config import get_artemis_config_value

ARTEMIS_LOGGER = logging.getLogger('artemis')


class EventSetException(Exception):
    pass

def am_I_in_slurm_environment():
    sub = subprocess.Popen("which srun", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
    return len(sub.stdout.read()) != 0

def one_time_send_to(address, port, message):
    '''
    Send message to given address at port. This method is not intended to be used repeatedly at the same port as it does not maintain a socket connection

    :param address: A string IP address to send to.
    :param port: An integer port number, or a string which can be mapped to an integeer
    :param message: A string to send
    '''

    server_address = (address, int(port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(tuple(server_address))
    except:
        raise

    send_size(sock, message)


def queue_to_host(queue, return_address, return_port, termination_event):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((return_address, return_port))
    except:
        sys.stderr.write("Error in connecting to port %s on address %s\n"%(return_port,return_address))
        sys.stderr.flush()
        raise
    while True:
        if termination_event.is_set() and queue.empty():
            break
        try:
            pickled_result = queue.get(timeout=1)
        except Queue.Empty:
            pass
        else:
            send_size(sock, pickled_result)
    sock.close()



def get_socket(address, port):
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
        except socketserver.socket.error as exc:
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
    except socketserver.socket.error as exc:
        raise

def wrap_queue_get_with_event_and_timeout(input_queue, event, timeout):
    '''
    This method wraps the given queue's get method, waiting at most timeout seconds before throwing a queue.Empty.
    Should the given event be set before the queue returned something, this method will return None.
    :param queue:
    :param event:
    :param timeout:
    :return:
    '''
    start = time.time()
    if timeout is not None:
        int_timeout = timeout if timeout < 0.1 else 0.1
    else:
        int_timeout = 0.1
    while timeout is None or (time.time() - start <= timeout):
        try:
            serialized_out = input_queue.get(timeout=int_timeout)
            return serialized_out
        except Queue.Empty:
            pass
        if event.is_set():
            raise EventSetException
    raise Queue.Empty


def recv_bytes(sock, size):
    buf = b''
    while size:
        try:
            newbuf = sock.recv(size)
        except socketserver.socket.error as exc:
            if exc.args[0] == 54:
                print("Connection reset by peer", file=sys.stderr)
                sys.exit(0)
            else:
                raise
        if not newbuf: break
        buf += newbuf
        size -= len(newbuf)
    return buf


def recv_size(sock,timeout=None):
    t_start = time.time()
    while True:
        size_data = recv_bytes(sock, 4)
        if size_data:
            size = struct.unpack('!I', size_data)[0]
            message = recv_bytes(sock, size)
            break
        elif timeout is not None and time.time() - t_start > timeout:
            raise socket.timeout
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
    if not isinstance(port,str):
        port = str(port)
    port_pattern = re.compile("^([0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])$")
    return port_pattern.match(port)


def get_remote_artemis_path(remote_ip):
    return get_artemis_config_value(
        section=remote_ip,
        option='artemis_path',
        default_generator=lambda: input('Specify Remote Artemis Installation Path: ').strip(),
        write_default=True
        )


def check_if_port_is_free(ip_address, port):
    '''
    This checks if the remote server is able to accept requests at the given port.
    :return:
    '''

    try:
        port = int(port)
    except ValueError:
        print ("Please provide a valid port for address %s. Received %s instead" %(ip_address, port))
        raise

    if ip_address in get_local_ips():
        import socket
        s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((ip_address,port))
        except:
            raise
        finally:
            s.close()
    else:
        check_ssh_connection(ip_address)
        ssh_connect = get_ssh_connection(ip_address=ip_address)
        check_port_function = 'python -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM);s.bind((\'%s\',%i));s.close()"'%(ip_address,port)
        stdin , stdout, stderr = ssh_connect.exec_command(check_port_function)
        err = stderr.read()
        assert not err, "The remote address %s cannot allocate port %i. The following error was raised: \n %s" % (ip_address, port,err.strip().split("\n")[-1])


def get_ssh_connection(ip_address):
    '''
    This returns a ssh_connection to the given ip_address. Make sure to close the connection afterwards.
    Requires a public/private key to be set up with the remote system. The location of the private key can be
    specified in .artemisrc or, if not specified, will be looked for in ~/.ssh/id_rsa
    :param ip_address:
    :return:
    '''
    import paramiko
    path_to_private_key = os.path.join(os.path.expanduser("~"),".ssh/id_rsa")
    private_key = paramiko.RSAKey.from_private_key_file(os.path.expanduser(path_to_private_key))
    username = get_artemis_config_value(section=ip_address, option="username", default_generator=lambda: getpass.getuser())
    ssh_conn = paramiko.SSHClient()
    ssh_conn.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_conn.connect(hostname=ip_address, username=username, pkey=private_key)
    return ssh_conn


def check_ssh_connection(ip_address):
    '''
    tries to load necessary information from the remote ~/.artemisrc file and execute a test remote function call. This is to verify that ssh connection is available
    :param ip_address: the ip_address to call against
    :return:
    '''

    test_function = 'python -c "import socket; print([l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith(\'127.\')][:1], [[(s.connect((\'8.8.8.8\', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0])"'
    ssh_conn = get_ssh_connection(ip_address=ip_address)
    stdin , stdout, stderr = ssh_conn.exec_command(test_function)
    out = stdout.read().strip()
    assert out == ip_address, "The remote server resolved a different ip-address than the one this computer used to contact it. This must not be a problem, but may be worth investigating"
    err = stderr.read()
    assert not err, "The remote server could not execute the test function. It returned the following error: \n %s"%err
    ssh_conn.close()

def check_pid(pid):
    """ Check For the existence of a unix pid.
    Source: https://stackoverflow.com/a/568285/2068168"""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True
