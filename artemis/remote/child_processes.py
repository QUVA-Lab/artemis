from __future__ import print_function

from six.moves import queue
from six import string_types
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
import pickle
import pipes
import logging

from artemis.config import get_artemis_config_value
from artemis.general.functional import get_partial_root
from artemis.general.should_be_builtins import file_path_to_absolute_module
from artemis.remote import remote_function_run_script, remote_generator_run_script
from artemis.remote.plotting.utils import handle_socket_accepts

from artemis.remote.utils import get_local_ips, get_socket, get_ssh_connection, check_pid, wrap_queue_get_with_event_and_timeout, EventSetException

ARTEMIS_LOGGER = logging.getLogger('artemis')

HEART_BEAT_FREQUENCY = 1

class ChildProcess(object):
    '''
    Generic Child Process
    '''
    counter=1
    def __init__(self, command, ip_address = 'localhost', name=None, take_care_of_deconstruct=False, set_up_port_for_structured_back_communication=False, termination_event=None,daemonize=False):
        '''
        Creates a ChildProcess
        :param ip_address: The command will be executed at this ip_address
        :param command: the command to execute.
        :param name: optional name. If not set, will be process_i, with i a global counter
        :param take_care_of_deconstruct: If set to True, deconstruct() is registered at exit
        :param port_for_structured_back_communication: Needs to be implemented according to the properties of the child process (see PythonChildProcess)
        :param termination_event: If passed, this is event is used - otherwise an event specific to this CP is created (anc can be accessed later)
        :param daemonize: If set to true, the stdout and stderr forwarding threads are daemonized. This is necessary if the childprocess
        is not expected to terminate on its own and instead, should terminate when the main thread terminates.
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
        self._execution_terminated_event = threading.Event() if termination_event is None else termination_event
        self.command = command
        self.id = uuid.uuid4()
        self.channel = None
        self.cp_started = False
        self.take_care_of_deconstruct = take_care_of_deconstruct
        self._sub_threads = []
        self._daemonize = daemonize
        if self.take_care_of_deconstruct:
            atexit.register(self.deconstruct)

    def is_daemonized(self):
        return self._daemonize

    def join_sub_threads(self):
        for th in self._sub_threads:
            i = 0
            while True:
                th.join(1.0)
                if th.is_alive():
                    i+=1
                    if i % 10==0:
                        ARTEMIS_LOGGER.warn("Still waiting for thread %s to join"%(th.name))
                else:
                    break

    def get_termination_event(self):
        ''' This should be read_only! '''
        return self._execution_terminated_event

    def set_termination_event(self,event):
        self._execution_terminated_event = event

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

    def deconstruct(self, signum=signal.SIGINT, system_signal=False):
        '''
        This completely and safely deconstructs a remote connection. It might also be called at program shutdown, if take_care_of_deconstruct is set to True
        kills itself if alive, then closes remote connection if applicable
        :return:
        '''
        if self.cp_started:
            self.get_termination_event().set()
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
            return self.is_alive()
        else:
            return False

    def is_local(self):
        return self.local_process

    def get_name(self):
        return self.name

    def get_ip(self):
        return self.ip_address

    def get_id(self):
        return self.ip_address + "_" + str(self.id)

    def _assign_pid(self,pid):
        try:
            int(pid)
        except:
            raise
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
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.cp_started = True
        t = threading.Thread(target=self._monitore_heart_beat,args=(self.get_termination_event(),),name="%s_HeartBeat"%self.get_name())
        if self.is_daemonized():
            t.setDaemon(True)
        self._sub_threads.append(t)
        t.start()
        return (stdin, stdout, stderr)

    def _monitore_heart_beat(self,event):
        # time.sleep(2)
        def wait_interrupt():
            start = time.time()
            while time.time() - start <= HEART_BEAT_FREQUENCY:
                # event = threading.Event()
                if event.wait(0.1):
                    return True
                time.sleep(0.1)
            return False

        while True:
            if not self.is_alive():
                ARTEMIS_LOGGER.info("%s >> heart has stopped; Event was%s" % (self.get_name()," set" if event.is_set() else " not set"))
                event.set()
                break
            else:
                res = wait_interrupt()
                if res:
                    ARTEMIS_LOGGER.info("%s >> Termination Event Set before detecting own death. Stopping Heartbeat"%self.get_name())
                    is_still_alive = self.deconstruct()
                    # if is_still_alive:
                    #     print("Deconstructing did not kill the process")
                    # else:
                    #     print("Deconstruct successfull")
                    break


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
            assert isinstance(command,string_types)
            command = "echo $$ ; exec %s" % command
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
            ARTEMIS_LOGGER.info("Not started yet, no kill command will be sent")
            return
        self.get_termination_event().set()
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
            res = self.sub.poll()
            alive = res == None
            return alive
        else:
            if self.ssh_conn.get_transport() is None:
                return False
            else:
                alive = self.ssh_conn.get_transport().is_active()
                if alive is False:
                    return False
                else:
                    command = "ps -h -p %s"%self.get_pid()
                    _,_,stdout,_ = self._run_command(command)
                    out = stdout.read()
                    pid = self.get_pid()
                    alive = pid in out
                    return alive


def listen_on_port(port=7000,name=None, received_termination_event=None):
    '''
    Sets up a thread that listens on a port forwards communication to a queue. Both, Queue and the port it listens on are returned
    :return: queue, port
    '''
    sock, port = get_socket("0.0.0.0", port=port)
    main_input_queue = queue.Queue()
    t = threading.Thread(target=handle_socket_accepts,args=(sock, main_input_queue, None,1,"%s_handle_socket_accepts"%name,10,received_termination_event),name=name)
    t.setDaemon(True)
    t.start()
    return main_input_queue, port, t


class RemotePythonProcess(ChildProcess):
    """
    Launch a python child process.
    """

    def __init__(self, function, ip_address,set_up_port_for_structured_back_communication=True, **kwargs):
        if ip_address=='localhost':
            ip_address="127.0.0.1"

        pickled_function = pickle_dumps_without_main_refs(function)
        self.encoded_pickled_function = base64.b64encode(pickled_function)
        self.is_generator = inspect.isgeneratorfunction(get_partial_root(function))
        remote_run_script = remote_generator_run_script if self.is_generator else remote_function_run_script
        remote_run_script_path = inspect.getfile(remote_run_script)
        if remote_run_script_path.endswith('pyc'):
            remote_run_script_path = remote_run_script_path[:-1]
        self.remote_run_script_path = remote_run_script_path

        command = "PLACEHOLDER"
        super(RemotePythonProcess, self).__init__(ip_address=ip_address, command=command, set_up_port_for_structured_back_communication=set_up_port_for_structured_back_communication, **kwargs)

    def prepare_command(self,command):
        # assert self.is_local(), "This, for now, assumes local execution"
        if self.set_up_port_for_structured_back_communication:
            self.return_value_queue, return_port, t = listen_on_port(7000, "%s_listen_on_port"%self.get_name(),received_termination_event=self.get_termination_event())
            self._sub_threads.append(t)
        else:
            return_port = -1
        all_local_ips = get_local_ips()
        return_address = all_local_ips[-1]
        ARTEMIS_LOGGER.info("Accepting requests to port %s on address %s" % (return_port, return_address))
        python_executable = get_artemis_config_value(section=self.get_ip(), option="python", default_generator=lambda: sys.executable)
        remote_script_path = apply_path_mapping(self.get_ip(),self.remote_run_script_path)

        command = [python_executable, '-u',remote_script_path, self.encoded_pickled_function, return_address, str(return_port)]
        if self.is_local():
            return command
        else:
            return " ".join(command)

    def get_return_queue(self):
        assert self.cp_started, "Not started yet, queue not yet set up"
        assert self.set_up_port_for_structured_back_communication, '{} has not been set up to send back a return value.'.format(self)
        return self.return_value_queue

    def get_return_value(self, timeout=None):
        assert self.set_up_port_for_structured_back_communication, '{} has not been set up to send back a return value.'.format(self)
        assert not self.is_generator, "The remotely executed function yields, it does not return a value. Use get_return_generator()"
        assert self.cp_started, "This ChildProcess has not been started yet"
        try:
            out_message = wrap_queue_get_with_event_and_timeout(self.return_value_queue, self.get_termination_event(), timeout=timeout)
            serialized_out = out_message.dbplot_message
        except Exception as e:
            if isinstance(e, EventSetException):
                return None
            elif isinstance(e, queue.Empty):
                ARTEMIS_LOGGER.warn(
                    "%s >> Child Process did not return a result in %s seconds and timed out. Exiting return generator" % (self.get_name(), timeout))
            else:
                ARTEMIS_LOGGER.warn("%s >> Child Process received an Exception. It is, at the moment, %s and the termination event is %s" % (
                    self.get_name(), "alive" if self.is_alive() else "dead", "set" if self.get_termination_event().is_set() else "not set"))
            raise
        res = pickle.loads(serialized_out)
        self.get_termination_event().set()
        self.join_sub_threads()
        return res


    def get_return_generator(self,timeout=None):
        assert self.is_generator, "The remotely executed function does not yield, it returns. Use get_return_value()"
        assert self.set_up_port_for_structured_back_communication, '{} has not been set up to send back a return value.'.format(self)
        while True:
            try:
                out_message = wrap_queue_get_with_event_and_timeout(self.return_value_queue,self.get_termination_event(),timeout=timeout)
                serialized_out = out_message.dbplot_message
                res = pickle.loads(serialized_out)
            except Exception as e:
                if isinstance(e,EventSetException):
                    ARTEMIS_LOGGER.info("%s >> Termination Event set, exiting return generator"%(self.get_name()))
                elif isinstance(e, queue.Empty):
                    ARTEMIS_LOGGER.warn("%s >> Child Process did not yield a result in %s seconds and timed out. Exiting return generator" % (self.get_name(), timeout))
                else:
                    ARTEMIS_LOGGER.warn("%s >> Child Process received an Exception. It is, at the moment, %s and the termination event is %s" % (
                        self.get_name(), "alive" if self.is_alive() else "dead", "set" if self.get_termination_event().is_set() else "not set"))
                res = StopIteration

            if res == StopIteration:
                self.get_termination_event().set()
                self.join_sub_threads()
                raise StopIteration
            yield res



class SlurmPythonProcess(RemotePythonProcess):
    def __init__(self, function, ip_address, set_up_port_for_structured_back_communication=True, slurm_kwargs={}, slurm_command="srun", **kwargs):
        '''

        :param function:
        :param ip_address:
        :param set_up_port_for_structured_back_communication:
        :param slurm_kwargs:
        :param kwargs:
        '''
        if ip_address=='localhost':
            ip_address="127.0.0.1"
        local_ips = get_local_ips()
        assert ip_address in local_ips, "At the moment, we want you to start a slurm process only from localhost"
        assert slurm_command in ["srun"], "At the moment, we only support 'srun' for execution of slurm"
        super(SlurmPythonProcess,self).__init__(function, ip_address, set_up_port_for_structured_back_communication, **kwargs)
        self.slurm_kwargs = slurm_kwargs


    def prepare_command(self,command):
        '''
        All the stuff that I need to prepare for the command to definitely work
        :param command:
        :return:
        '''

        command = super(SlurmPythonProcess,self).prepare_command(command)
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

    def kill(self,signum=signal.SIGINT):
        # Slurm needs the signal twice
        if not self.cp_started:
            ARTEMIS_LOGGER.info("Not started yet, no kill command will be sent")
            return
        if self.is_local():
            if check_pid(self.sub.pid):
                self.sub.send_signal(signum)
                time.sleep(0.15)
                self.sub.send_signal(signum)
        else:
            raise NotImplementedError()

def apply_path_mapping(ip_address, path):
    path_mapping = get_artemis_config_value(section=ip_address,option="path_mapping",default_generator=lambda : None)
    if path_mapping is not None:
        to_replace,replace_with = path_mapping.split(";")
        if to_replace in path:
            return path.replace(to_replace,replace_with)
    return path


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
    except :
        ARTEMIS_LOGGER.info("Using Dill")
        # TODO: @petered There is something very fishy going on here that I don't understand.
        import dill
        pickle_str = dill.dumps(obj, protocol=0)

    pickle_str = pickle_str.replace('__main__', module_path)  # Hack!
    return pickle_str


def pickle_dump_without_main_refs(obj, file_obj):
    string = pickle_dumps_without_main_refs(obj)
    file_obj.write(string)
