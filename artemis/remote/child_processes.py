from __future__ import print_function

from six.moves import queue
import atexit
import base64
import inspect
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

import os
import pickle
import pipes


from six import string_types

from artemis.general.should_be_builtins import file_path_to_absolute_module
from artemis.remote import remote_function_run_script
from artemis.remote.plotting.utils import handle_socket_accepts

from artemis.config import get_artemis_config_value
from artemis.remote.utils import get_local_ips, get_socket, get_ssh_connection, check_pid


class ChildProcess(object):
    '''
    Generic Child Process
    '''
    counter=1
    def __init__(self, command, ip_address = 'localhost', name=None, take_care_of_deconstruct=False, set_up_port_for_structured_back_communication=False):
        '''
        Creates a ChildProcess
        :param ip_address: The command will be executed at this ip_address
        :param command: the command to execute.
        :param name: optional name. If not set, will be process_i, with i a global counter
        :param take_care_of_deconstruct: If set to True, deconstruct() is registered at exit
        :param port_for_structured_back_communication: Needs to be implemented according to the properties of the child process (see PythonChildProcess)
        :return:
        '''
        if name is None:
            name = "process_%s"%ChildProcess.counter
        if ip_address=='localhost':
            ip_address='127.0.0.1'
        ChildProcess.counter += 1
        self.name = name
        self.ip_address = ip_address
        self.local_process = self.ip_address in get_local_ips()
        self.set_up_port_for_structured_back_communication = set_up_port_for_structured_back_communication
        self.queue_from_cp = None

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
            return command
        else:
            if isinstance(command,list):
                command = " ".join(pipes.quote(c) for c in command)

            return self.get_extended_command(command)
        return command

    def get_extended_command(self,command):
        return "echo $$ ; exec %s"%command

    def deconstruct(self, signum=signal.SIGKILL, system_signal=False):
        '''
        This completely and safely deconstructs a remote connection. It might also be called at program shutdown, if take_care_of_deconstruct is set to True
        kills itself if alive, then closes remote connection if applicable
        :return:
        '''
        if self.cp_started:
            if signum == signal.SIGKILL:
                self.kill(signum=signum)
            elif signum == signal.SIGINT:
                self.kill(signum=signum)
                counter = 0
                while counter < 3:
                    if self.is_alive():
                        # 3 seconds grace period
                        counter +=1
                        time.sleep(1.0)
                    else:
                        break
                if self.is_alive():
                    # grace perior exceeded, terminating
                    self.kill(signum=signal.SIGTERM)


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

    def get_queue_from_cp(self):
        assert self.set_up_port_for_structured_back_communication, "You did not specify a port to be set up. Don't expect structured info from child process"
        assert self.queue_from_cp, "The queue has not been set up yet. Did you start the child process before calling this function? Also make sure your implementation of child process has this functionality"
        return self.queue_from_cp

    def execute_child_process(self):
        '''
        Executes ChildProcess in a non-blocking manner. This returns immediately.
        This method returns a tuple (stdin, stdout, stderr) of the child process
        :return:
        '''
        if not self.is_local():
            self.ssh_conn = get_ssh_connection(self.ip_address)
        command = self.prepare_command(self.command)
        pid, stdin, stdout, stderr = self._run_command(command,get_pty=True)
        self._assign_pid(pid)
        self.cp_started = True
        return (stdin, stdout, stderr)

    def _run_command(self,command,get_pty=False):
        '''
        execute the given command
        :param command: string, to execute
        :return: (stdin, stdout, stderr)
        '''
        if self.local_process:
            if type(command) == list:
                sub = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elif isinstance(command, string_types):
                shlexed_command = shlex.split(command)
                sub = subprocess.Popen(shlexed_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                raise NotImplementedError()
            self.sub = sub
            stdin = sub.stdin
            stdout = sub.stdout
            stderr = sub.stderr
            pid = sub.pid
        else:
            stdin, stdout, stderr = self.ssh_conn.exec_command(command,get_pty=get_pty)
            pid = stdout.readline().strip()
        return (pid, stdin, stdout, stderr)

    def kill(self, signum=signal.SIGINT):
        '''
        Kills the process by
        (remote) sending 'kill -s signal pid' to the server.
        (local)  os.kill(pid, signal)
        default signal is SIGINT
        This call does not block. The success of killing the process
        needs to be determined by the user. E.g by calling is_alive().
        In case the process was killed, the ssh connection is terminated (if remote)
        :return:
        '''

        if not self.cp_started:
            print("Not started yet, no kill command will be sent")
            return
        if self.is_local():
            if check_pid(self.sub.pid):
                self.sub.send_signal(signum)
        else:
            if self.ssh_conn.get_transport() is None:
                return
            kill_command = "kill -s %s %s" %(signum, self.get_pid())
            self._run_command(kill_command)
            if self.ssh_conn.get_transport() is not None:
                if not self.is_alive() and not self.is_local():
                    self.ssh_conn.close()


    def is_alive(self):
        if not self.cp_started:
            return False
        if self.is_local():
            return self.sub.poll() == None
        else:
            if self.ssh_conn.get_transport() is None:
                return False
            else:
                command = "echo $$; exec ps -h -p %s"%self.get_pid()
                _,_,stdout,_ = self._run_command(command)
                return self.get_pid() in stdout.read()


class PythonChildProcess(ChildProcess):
    '''
    This ChildProcess is designed to spawn python processes.
    '''
    def __init__(self, ip_address, command, **kwargs):
        '''
        Creates a PythonChildProcess
        :param ip_address: The command will be executed at this ip_address
        :param command: the command to execute. Is assumed to be a python call
        :param name: optional name. If not set, will be process_i, with i a global counter
        :param take_care_of_deconstruct: If set to True, deconstruct() is registered at exit
        :param port_for_structured_back_communication: If set, the ip-address of this device as well as a specific port will be appended as arguments to the executed python command in the format --port=1234 --address=127.0.0.1
        The child process is responsible for reading these values out and communicating to it. If set, this child process will expose a queue on that address with get_queue_from_cp()
        :return:
        '''
        super(PythonChildProcess,self).__init__(ip_address=ip_address, command=command,**kwargs)

    def prepare_command(self,command):
        '''
        All the stuff that I need to prepare for the command to definitely work
        :param command:
        :return:
        '''
        if self.set_up_port_for_structured_back_communication:
            self.queue_from_cp, port = listen_on_port()
            if self.is_local():
                address = "127.0.0.1"
            else:
                address = get_local_ips()[-1]

        if self.is_local():
            home_dir = os.path.expanduser("~")
        else:
            _,_,stdout,_ = self._run_command("echo $$; exec echo ~")
            home_dir = stdout.read().strip()

        if type(command) == list:
            if self.set_up_port_for_structured_back_communication:
                command.append("--port=%i"%port)
                command.append("--address=%s"%address)
            if not self.local_process:
                command = [c.replace("python", self.get_extended_command(get_artemis_config_value(section=self.get_ip(), option="python", default_generator=lambda: sys.executable)), 1) if c.startswith("python") else c for c in command]
                command = [s.replace("~",home_dir) for s in command]
                command = " ".join([c for c in command])
            else:
                command = [c.strip("'") for c in command]
                command = [c.replace("python", sys.executable, 1) if c.startswith("python") else c for c in command]
                command = [s.replace("~",home_dir) for s in command]

        elif isinstance(command, string_types) and command.startswith("python"):
            if self.set_up_port_for_structured_back_communication:
                command += " --port=%i "%port
                command += "--address=%s"%address
            if not self.local_process:
                command = command.replace("python", self.get_extended_command(get_artemis_config_value(section=self.get_ip(), option="python",default_generator=lambda: sys.executable)), 1)
            else:
                command = command.replace("python", sys.executable)
            command = command.replace("~",home_dir)
        else:
            raise NotImplementedError()
        return command


def listen_on_port(port=7000):
    '''
    Sets up a thread that listens on a port forwards communication to a queue. Both, Queue and the port it listens on are returned
    :return: queue, port
    '''
    sock, port = get_socket("0.0.0.0", port=port)
    sock.listen(1)
    main_input_queue = queue.Queue()
    t = threading.Thread(target=handle_socket_accepts,args=(sock, main_input_queue, None,1))
    t.setDaemon(True)
    t.start()
    return main_input_queue, port


class RemotePythonProcess(ChildProcess):
    """
    Launch a python child process.
    """

    def __init__(self, function, ip_address, set_up_port_for_structured_back_communication=True, **kwargs):
        if ip_address=='localhost':
            ip_address="127.0.0.1"

        pickled_function = pickle_dumps_without_main_refs(function)
        encoded_pickled_function = base64.b64encode(pickled_function)

        remote_run_script_path = inspect.getfile(remote_function_run_script)
        if remote_run_script_path.endswith('pyc'):
            remote_run_script_path = remote_run_script_path[:-1]

        self.return_value_queue, return_port = listen_on_port(7000)
        all_local_ips = get_local_ips()
        return_address = "127.0.0.1" if ip_address in all_local_ips else all_local_ips[-1]
        command = [sys.executable, '-u', remote_run_script_path, encoded_pickled_function, return_address, str(return_port)]
        super(RemotePythonProcess, self).__init__(ip_address=ip_address, command=command, set_up_port_for_structured_back_communication=set_up_port_for_structured_back_communication,  **kwargs)

    def get_return_value(self, timeout=1):
        assert self.set_up_port_for_structured_back_communication, '{} has not been set up to send back a return value.'.format(self)
        serialized_out = self.return_value_queue.get(timeout=timeout)
        out = pickle.loads(serialized_out.dbplot_message)
        return out

class SlurmPythonProcess(RemotePythonProcess):
    def __init__(self, function, ip_address, set_up_port_for_structured_back_communication=True, slurm_kwargs={}, slurm_command="srun", **kwargs):
        '''

        :param function:
        :param ip_address:
        :param set_up_port_for_structured_back_communication:
        :param slurm_kwargs:
        :param kwargs:
        '''
        assert ip_address in get_local_ips(), "At the moment, we want you to start a slurm process only from localhost"
        assert slurm_command in ["srun"], "At the moment, we only support 'srun' for execution of slurm"
        super(SlurmPythonProcess,self).__init__(function, ip_address, set_up_port_for_structured_back_communication, **kwargs)
        self.slurm_kwargs = slurm_kwargs


    def prepare_command(self,command):
        '''
        All the stuff that I need to prepare for the command to definitely work
        :param command:
        :return:
        '''

        slurm_command = "srun"
        for k,v in self.slurm_kwargs.items():
            if k.startswith("--"):
                slurm_command += " %s=%s"%(k,v)
            elif k.startswith("-"):
                slurm_command += " %s %s"%(k,v)

        if isinstance(command, list):
            command = " ".join(pipes.quote(c) for c in command)

        final_command = " ".join((slurm_command,command))
        if self.is_local():
            return final_command
        else:
            raise NotImplementedError()

def pickle_dumps_without_main_refs(obj):
    """
    Yeah this is horrible, but it allows you to pickle an object in the main module so that it can be reloaded in another
    module.
    :param obj:
    :return:
    """
    currently_run_file = sys.argv[0]
    module_path = file_path_to_absolute_module(currently_run_file)
    try:
        pickle_str = pickle.dumps(obj, protocol=0)
    except:
        print("Using Dill")
        # TODO: @petered There is something very fishy going on here that I don't understand.
        import dill
        pickle_str = dill.dumps(obj, protocol=0)

    pickle_str = pickle_str.replace('__main__', module_path)  # Hack!
    return pickle_str


def pickle_dump_without_main_refs(obj, file_obj):
    string = pickle_dumps_without_main_refs(obj)
    file_obj.write(string)
