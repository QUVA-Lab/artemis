import shlex
import subprocess
import sys
import threading

from artemis.remote.utils import get_local_ips, get_ssh_connection




def execute_command(ip_address, command, blocking=True):
    '''
    This method spawns a child-process (either locally or remote, depending on the ip_address). Then it executes the given command and handles communication.
    If blocking is True, then this call does not return before the child process returns. It then prints stdout to console, followed by stderr.
    If blocking is False, then this call returns immediately and asynchronously forwards stdout and stderr to the console in separate threads.
    If ip_address is local, then the command will be split using shlex.split() and must be formatted accordingly. The subprocess call is executed with shell=False
    for all the right reasons.
    :param ip_address:
    :param command: String. command to execute
    :param blocking:
    :return:
    '''
    #TODO: Reformat this to work with the Nanny. Let it manage the communication
    if ip_address in get_local_ips():
        sub = subprocess.Popen(shlex.split(command),stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        stdout_pipe = sub.stdout
        stderr_pipe = sub.stderr

    else:
        ssh_conn = get_ssh_connection(ip_address)
        transport = ssh_conn.get_transport()
        channel = transport.open_session()
        channel.exec_command(command)#
        bufsize=-1
        stdout_pipe = channel.makefile('r', bufsize)
        stderr_pipe = channel.makefile_stderr('r',bufsize)

    #stdout
    t1 = ParamikoPrintThread(source_pipe=stdout_pipe, target_pipe=sys.stdout)
    t1.start()
    # stderr
    t2 = ParamikoPrintThread(source_pipe=stderr_pipe, target_pipe=sys.stderr)
    t2.start()
    if blocking:
        t1.join()
        t2.join()

class ParamikoPrintThread(threading.Thread):
    def __init__(self, source_pipe, target_pipe, stopping_criterium=None, stop_event=None, prefix=""):
        '''
        :param source_pipe: The ssh pipe from which to forward communications
        :param target_pipe: Either stdout or stderr. This determines if stderr or stdout is read from the ssh channel.
        :param stopping_criterium: function which takes in a line from source_pipe and evaluates to boolean.:
        :param prefix: A prefix that is attached to every printed line.
        :return:
        '''
        self.source_pipe = source_pipe
        self.prefix = prefix
        self.target_pipe = target_pipe
        self.stopping_criterium = stopping_criterium
        super(ParamikoPrintThread, self).__init__()

    def run(self, ):
        with self.source_pipe:
            # Terminates when source pipe runs dry.
            for line in iter(self.source_pipe.readline, b''):
                self.target_pipe.write("%s%s" % (self.prefix, line))
                self.target_pipe.flush()
                if self.stopping_criterium is not None and self.stopping_criterium(line):
                    break