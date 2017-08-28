from artemis.config import check_or_create_artemis_config, get_artemis_config_value
from artemis.fileman.config_files import get_config_sections


def validip(ip):
    return ip.count('.') == 3 and  all(0<=int(num)<256 for num in ip.rstrip().split('.'))


def get_configured_machines():
    """
    :return: A dict<machine_name: {'ip': ip_address, 'username': username}>
    """
    config_filename = check_or_create_artemis_config()
    sections = get_config_sections(config_filename)
    ip_sections = [s for s in sections if s.startswith('remote:')]
    machines = {}
    for sec in ip_sections:
        machine_name = sec[len('remote:'):]
        ip = get_artemis_config_value(section=sec, option='ip')
        username = get_artemis_config_value(section=sec, option='username')
        machines[machine_name] = {'ip': ip, 'username': username}
    return machines


EXAMPLE_MACHINE = """[remote:my_machine1]
ip=123.456.789.01
username=pluto
"""


def get_remote_machine_info(machine_name):
    machines = get_configured_machines()
    if len(machines)==0:
        raise Exception('No machines are specified!, You can add machines to ~/.artemisrc, as \n{}'.format(EXAMPLE_MACHINE))
    if machine_name not in machines:
        raise Exception("Could not find '{}' in the list of machines: {}.  You can add it to ~/.artemisrc".format(machine_name, machines.keys()))
    return machines[machine_name]
