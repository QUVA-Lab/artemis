from __future__ import print_function

import Queue
import atexit
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

import os
from artemis.remote.plotting.utils import handle_socket_accepts

from artemis.config import get_artemis_config_value
from artemis.remote.utils import get_local_ips, get_socket, get_ssh_connection, check_pid


class ChildProcess(object):
    '''
    Generic Child Process
    '''
    counter=1
    def __init__(self, ip_address, command, name=None, take_care_of_deconstruct=False, set_up_port_for_structured_back_communication=False):
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
            return self.get_extended_command(command)
        return command

    def get_extended_command(self,command):
        return "echo $$ ; exec %s"%command

    def deconstruct(self, signum=signal.SIGINT, system_signal=False):
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
            elif type(command) == str or type(command) == unicode:
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
    def __init__(self,*args,**kwargs):
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
        super(PythonChildProcess,self).__init__(*args,**kwargs)

    def listen_on_port(self):
        '''
        Sets up a thread that listens on a port forwards communication to a queue. Both, Queue and the port it listens on are returned
        :return: queue, port
        '''
        sock, port = get_socket("0.0.0.0", port=7000)
        sock.listen(1)
        main_input_queue = Queue.Queue()
        t = threading.Thread(target=handle_socket_accepts,args=(sock, main_input_queue, None,1))
        t.setDaemon(True)
        t.start()
        return main_input_queue, port

    def prepare_command(self,command):
        '''
        All the stuff that I need to prepare for the command to definitely work
        :param command:
        :return:
        '''
        if self.set_up_port_for_structured_back_communication:
            self.queue_from_cp, port = self.listen_on_port()
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

        elif type(command) == str or type(command) == unicode and command.startswith("python"):
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
