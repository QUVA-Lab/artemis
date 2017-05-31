from __future__ import print_function

import atexit
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from ConfigParser import NoOptionError
import paramiko
from artemis.config import get_artemis_config_value
from artemis.fileman.config_files import get_config_value
from artemis.remote.utils import get_local_ips


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
                    self.target_pipe.write("%s%s"%(self.prefix,line))
                    self.target_pipe.flush()
                    if self.stopping_criterium is not None and self.stopping_criterium(line):
                        break


class Nanny(object):
    '''
    Manages child processes. This class manages the start, live and deconstruction of child processes across different machines.
    '''
    def __init__(self):
        self.child_processes = {}
        self.stdout_threads = {}
        self.original_sigint_handler = signal.getsignal(signal.SIGINT)
        self.original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self.deconstruct)
        signal.signal(signal.SIGTERM, self.deconstruct)
        # atexit.register(self.deconstruct)

    def register_child_process(self, cp):
        '''
        This adds a child process to the Nanny.
        :return:
        '''
        self.child_processes[cp.get_id()] = cp

    def execute_all_child_processes(self, time_out=1, terminate_at_error=False,):
        '''
        Executes all child processes and starts managing communications. This method returns only when all child processes terminated.
        It might be the case that some child-processes hang or don't terminate. In this case, when one (the first) process
          terminates, all other processes are killed after time_out seconds. This behaviour is triggered also when

        :param: time_out
        :return:
        '''

        termination_request_event = threading.Event()
        stdout_stopping_criterium = lambda x: "Training finished" in x
        stdout_threads = {}
        stderr_threads = {}

        N_workers = 0
        def err_fun(cp_ip):
            print("Error in process on %s:"%cp_ip)
            return True
        name_max_lenght = max([len(cp.name) for cp in self.child_processes.values()])
        for id, cp in self.child_processes.iteritems():
            stdin, stdout, stderr = cp.execute_child_process()

            prefix = cp.name.ljust(name_max_lenght)+": "
            if "worker" in cp.name:
                N_workers+=1
            stdout_thread = threading.Thread(target=self.monitor_and_forward_child_communication,
                                  args=(stdout,sys.stdout,termination_request_event,stdout_stopping_criterium, prefix))

            stderr_stopping_criterium = lambda x: err_fun(cp.get_ip()) if terminate_at_error else None
            stderr_thread = threading.Thread(target=self.monitor_and_forward_child_communication,
                                  args=(stderr,sys.stderr,termination_request_event,stderr_stopping_criterium, prefix))
            stdout_threads[cp.get_id()] = stdout_thread
            stderr_threads[cp.get_id()] = stderr_thread

            stdout_thread.start()
            stderr_thread.start()

        try:
            for _ in range(N_workers):
                while not termination_request_event.wait(0.01):
                    pass
                termination_request_event.clear()
        except KeyboardInterrupt:
            sys.exit(1)

        # Grace perior for other threads to shutdown
        time.sleep(time_out)
        for id,cp in self.child_processes.iteritems():
            if cp.is_alive():
                print("Child Process %s at %s did not terminate %s seconds after the first process in cluster terminated. Terminating now." %(cp.get_name(), cp.get_ip(), time_out))
                cp.kill()

        for id,cp in self.child_processes.iteritems():
            if cp.is_alive():
                print("Child Process %s at %s did not terminate. Force quitting now." %(cp.get_name(),cp.get_ip()))
                cp.deconstruct(signal.SIGTERM)

        # This should return immediatly, since the unterlying pipes should have run dry. if it doesn't I messed up...:
        for stdout_thread, stderr_thread in zip(stdout_threads.values(), stderr_threads.values()):
            assert not stdout_thread.is_alive(), "This should not have happened"
            assert not stderr_thread.is_alive(), "This should not have happened"

    def deconstruct(self, signum, frame=None):
        '''
        This method is called when SIGINT or SIGTERM are called.
        This aggressively deconstructs the Nanny and all child processes. Then, the signal is passed back to the original signal handlers.
        :return:
        '''

        for cp in self.child_processes.values():
            cp.kill()
        time.sleep(1.0)
        for cp in self.child_processes.values():
            if cp.is_alive():
                print("Child Process %s at %s still alive, force terminating now"% (cp.name, cp.get_ip()))
                cp.kill(signal=signal.SIGTERM)

        signal.signal(signal.SIGINT, self.original_sigint_handler)
        signal.signal(signal.SIGTERM, self.original_sigterm_handler)
        os.kill(os.getpid(), signum)

    def monitor_and_forward_child_communication(self, source_pipe, target_pipe, termination_request_event=None, stopping_criterium=None,  prefix=""):
        '''
        thread to forward communication from source_pipe to target_pipe
        :param source_pipe:
        :param target_pipe:
        :param termination_request_event: Is set once the source_pipe has closed and this thread terminates, or when ths stopping_criterium has been met.
        :param stopping_criterium:
        :param prefix:
        :return:
        '''
        with source_pipe:
            for line in iter(source_pipe.readline, b''):
                target_pipe.write("%s%s"%(prefix,line))
                target_pipe.flush()
                if stopping_criterium is not None and stopping_criterium(line) and not "pydev debugger" in line and line.strip():
                    target_pipe.write(source_pipe.read())
                    target_pipe.flush()
                    break
            if termination_request_event is not None:
                termination_request_event.set() # The input pipe closed, this thread terminates and we would like everybody to terminate


class ChildProcess(object):
    counter=1
    def __init__(self, ip_address, command, name=None, take_care_of_deconstruct=False):
        '''
        Creates a ChildProcess
        :param ip_address: The command will be executed at this ip_address
        :param command: the command to execute. If it is a python command and remote, will substitute the virtualenv
        :param name: optional name. If not set, will be process_i, with i a clobal counter
        :param take_care_of_deconstruct: If set to True, deconstruct() is registered at exit
        :return:
        '''
        if name is None:
            name = "process_%s"%ChildProcess.counter
        ChildProcess.counter += 1
        self.name = name
        self.ip_address = ip_address
        self.local_process = self.ip_address in get_local_ips()


        # command = command.replace(unichr(34),unichr(39))
        self.command = command
        self.id = uuid.uuid4()
        self.channel = None
        self.cp_started = False
        self.take_care_of_deconstruct = take_care_of_deconstruct
        if self.take_care_of_deconstruct:
            atexit.register(self.deconstruct)

    def prepare_command(self,command):
        '''
        All the stuff that I need to prepare for the command to definitely work
        :param command:
        :return:
        '''
        if self.is_local():
            home_dir = os.path.expanduser("~")

        else:
            _,_,stdout,_ = self._run_command("echo $$; exec echo ~")
            home_dir = stdout.read().strip()

        if type(command) == list:
            if not self.local_process:
                command = [c.replace("python", self.get_extended_command(get_artemis_config_value(section=self.get_ip(), option="python")), 1) if c.startswith("python") else c for c in command]
                command = [s.replace("~",home_dir) for s in command]
                command = " ".join([c for c in command])
            else:
                command = [c.strip("'") for c in command]
                command = [c.replace("python", sys.executable, 1) if c.startswith("python") else c for c in command]
                command = [s.replace("~",home_dir) for s in command]

        elif type(command) == str or type(command) == unicode and command.startswith("python"):
            if not self.local_process:
                command = command.replace("python", self.get_extended_command(get_artemis_config_value(section=self.get_ip(), option="python")), 1)
            else:
                command = command.replace("python", sys.executable)
            command = command.replace("~",home_dir)
        else:
            raise NotImplementedError()
        return command

    def get_extended_command(self,command):
        return "echo $$ ; exec %s"%command

    def deconstruct(self, message=signal.SIGINT):
        '''
        This completely and safely deconstructs a remote connection. It will also be called at program shutdown.
        kills itself if alive, then closes remote connection if applicable
        :return:
        '''
        print("Deconstructing {}".format(self.name))
        if not self.cp_started:
            # Nothing has happened yet
            return

        if self.is_alive():
            self.kill(signal=message)
            time.sleep(4.0)
        if self.is_alive():
            self.kill(signal.SIGTERM)
        if not self.is_local():
            self.ssh_conn.close()

    def is_local(self):
        return self.local_process

    def get_name(self):
        return self.name

    def get_ip(self):
        return self.ip_address

    def get_id(self):
        return self.ip_address + "_" + str(self.id)

    def _assign_pid(self,pid):
        self.pid = pid

    def get_pid(self):
        assert self.cp_started, "This ChildProcess has not been executed yet, no pid available yet. run execute_child_process() first"
        return self.pid

    def get_ssh_connection(self):
        assert not self.is_local(), "A local ChildProcess does not have a ssh_connection"
        assert self.cp_started, "A ssh_connection will only be available after running execute_child_process()"
        return self.ssh_conn

    def execute_child_process(self, get_pty = False):
        '''
        Executes ChildProcess in a non-blocking manner. This returns immediately.
        This method returns a tuple (stdin, stdout, stderr) of the child process
        :return:
        '''
        if not self.is_local():
            self.ssh_conn = get_ssh_connection(self.ip_address)
        command = self.prepare_command(self.command)
        pid, stdin, stdout, stderr = self._run_command(command, get_pty=False)
        self._assign_pid(pid)
        self.cp_started = True
        return (stdin, stdout, stderr)

    def _run_command(self,command, get_pty=False):
        '''
        execute the given command
        :param command: string, to execute
        :return: (stdin, stdout, stderr)
        '''
        if self.local_process:
            if type(command) == list:
                # print (" ".join(c for c in command))
                sub = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elif type(command) == str or type(command) == unicode:
                shlexed_command = shlex.split(command)
                sub = subprocess.Popen(shlexed_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # sub.communicate()
            else:
                raise NotImplementedError()
            self.sub = sub
            stdin = sub.stdin
            stdout = sub.stdout
            stderr = sub.stderr
            pid = sub.pid
        else:
            stdin, stdout, stderr = self.ssh_conn.exec_command(command)
            pid = stdout.readline().strip()
        return (pid, stdin, stdout, stderr)

    def kill(self, signal=signal.SIGINT):
        '''
        Kills the process by
        (remote) sending 'kill -s signal pid' to the server.
        (local)  os.kill(pid, signal)
        default signal is SIGINT
        This call does not block. The success of killing the process
        needs to be determined by the user. E.g by calling is_alive()
        :return:
        '''

        if not self.cp_started:
            print("Not started yet, no kill command will be sent")
            return
        if self.is_local():
            self.sub.send_signal(signal)
        else:
            kill_command = "kill -s %s %s" %(signal, self.get_pid())
            self._run_command(kill_command)

    def is_alive(self):
        if not self.cp_started:
            return False
        if self.is_local():
            return self.sub.poll() == None
        else:
            command = "echo $$; exec ps -h -p %s"%self.get_pid()
            _,_,stdout,_ = self._run_command(command)
            return self.get_pid() in stdout.read()


def get_ssh_connection(ip_address):
    '''
    This returns a ssh_connection to the given ip_address. Make sure to close the connection afterwards.
    Requires a public/private key to be set up with the remote system. The location of the private key can be
    specified in .artemisrc or, if not specified, will be looked for in ~/.ssh/id_rsa
    :param ip_address:
    :return:
    '''

    try:
        path_to_private_key = get_artemis_config_value(section=ip_address, option="private_key")
    except NoOptionError:
        path_to_private_key = os.path.join(os.path.expanduser("~"),".ssh/id_rsa")

    private_key = paramiko.RSAKey.from_private_key_file(os.path.expanduser(path_to_private_key))
    username = get_artemis_config_value(section=ip_address, option="username")
    ssh_conn = paramiko.SSHClient()
    ssh_conn.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_conn.connect(hostname=ip_address, username=username, pkey=private_key)
    return ssh_conn


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


def check_ssh_connection(ip_address):
    '''
    tries to load necessary information from the ~/.artemisrc file and execute a test remote function call. This is to verify that ssh connection is available
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


def check_if_port_is_free(ip_address, port):
    '''
    This checks if the remote server is able to accept requests at the given port.
    :param connections: A list ["ip_address:port"] of strings
    :param force_close: In case the port is in use, force close the program that is occupying it.
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
