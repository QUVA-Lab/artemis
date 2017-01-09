import socket

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