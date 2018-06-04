import atexit
import signal
import sys
import threading
import time

import logging

from artemis.general.generators import multiplex_generators
ARTEMIS_LOGGER = logging.getLogger('artemis')

#
# class ManagedChildProcess(object):
#     def __init__(self,cp,monitor_for_termination,monitor_if_stuck_timeout):
#         self.cp = cp
#         self.monitor_for_termination = monitor_for_termination
#         self.monitor_if_stuck_timeout = monitor_if_stuck_timeout
#
#     def get_process(self):
#         return self.cp
#
#     @property
#     def name(self):
#         return self.cp.name
#
#     def monitor_if_stuck(self):
#         return self.monitor_if_stuck_timeout is not None
#
#     def kill(self, signal=signal.SIGINT):
#         return self.cp.kill(signal)
#
#     def is_alive(self):
#         return self.cp.is_alive()
#
#     def get_name(self):
#         return self.cp.get_name()
#
#     def set_termination_event(self,event):
#         return self.cp.set_termination_event(event)
#
#     def get_termination_event(self):
#         return self.cp.get_termination_event()
#
#     def get_pid(self):
#         return self.cp.get_pid()
#
#     def get_ip(self):
#         return self.cp.get_ip()
#
#     def get_id(self):
#         return self.cp.get_id()
#
#     def execute(self):
#         return self.cp.execute_child_process()
#
#     def get_id(self):
#         return self.cp.get_id()
#
#     def deconstruct(self, signum=signal.SIGINT, system_signal=False):
#         return self.cp.deconstruct(signum)
from artemis.remote.child_processes import RemotePythonProcess


class Nanny(object):
    '''
    Manages child processes. This class manages the start, live and deconstruction of child processes across different machines.
    '''
    def __init__(self,name="",termination_event=None):
        self.managed_child_processes = {}
        self.stdout_threads = {}
        self.name = name
        self.termination_event = threading.Event() if termination_event is None else termination_event
        atexit.register(self.deconstruct)

    def register_child_process(self, cp, monitor_for_termination=True, monitor_if_stuck_timeout=None):
        '''
        This adds a child process to the Nanny.
        :param: monitor_for_termination: if set, this process' termination will trigger the shutdown process. If not set, this process' termination will go unnoticed
        :param: monitor_if_stuck_timeout: If set to a positive number, this process' communication is monitored. If not String has been passed over the Stdout channel of the child process in
         monitor_if_stuck_timeout seconds, the process is assumed to be stuck and the shutdown procedure is started.
        :return:
        '''
        # assert monitor_for_termination != False, "Not supported any more at the moment"
        assert monitor_if_stuck_timeout is None or (monitor_if_stuck_timeout > 0 and type(monitor_if_stuck_timeout) == int), "Please set monitor_if_stuck_timeout to a positive integer"
        cp.monitor_for_termination = monitor_for_termination
        cp.monitor_if_stuck_timeout = monitor_if_stuck_timeout
        self.managed_child_processes[cp.get_id()] = cp

    def get_child_processes(self):
        return {id:mcp.get_process() for id,mcp in self.managed_child_processes.items()}

    def multiplex_return_generators(self,time_out_kill=None):
        multiplex_gen = multiplex_generators([(cp.name,cp.get_return_generator(time_out_kill)) for cp in self.managed_child_processes.values()
                                     if cp.set_up_port_for_structured_back_communication and isinstance(cp,RemotePythonProcess) and cp.is_generator], stop_at_first=True)
        return multiplex_gen


    def execute_all_child_processes(self, stdout_stopping_criterium=lambda line: False, stderr_stopping_criterium=lambda line: False, block=True):
        '''
        :param stdout_stopping_criterium: a function that is evaluated per line of stdout output. If evaluates to True, the nanny is interrupted
        :param stderr_stopping_criterium:a function that is evaluated per line of stderr output. If evaluates to True, the nanny is interrupted
        :param block: Wether or not to bock this call untill all child processes have terminated or to immediately return
        :return:
        '''
        stdout_threads = {}
        stderr_threads = {}
        name_max_lenght = max([len(mcp.name) for mcp in self.managed_child_processes.values()])
        for i, id in enumerate(self.managed_child_processes.keys()):
            mcp =self.managed_child_processes[id]
            if mcp.monitor_for_termination:
                mcp.set_termination_event(self.termination_event)
            stdin, stdout, stderr = mcp.execute_child_process()

            prefix = mcp.name.ljust(name_max_lenght)+": "

            # True if in debug mode
            gettrace = getattr(sys, 'gettrace', None)
            monitor_if_stuck_timeout = mcp.monitor_if_stuck_timeout if not gettrace() else None # only set timeout if not in debug mode

            stdout_thread = threading.Thread(target=self._monitor_and_forward_child_communication,name="%s_stdout_Thread"%mcp.get_name(),
                                             args=(stdout, sys.stdout, mcp.name, mcp.get_termination_event(), stdout_stopping_criterium, prefix, monitor_if_stuck_timeout))

            stderr_thread = threading.Thread(target=self._monitor_and_forward_child_communication,name="%s_stderr_Thread"%mcp.get_name(),
                                             args=(stderr,sys.stderr,mcp.name,mcp.get_termination_event(),stderr_stopping_criterium, prefix, None))

            stdout_thread.setDaemon(True)
            stderr_thread.setDaemon(True)

            stdout_threads[mcp.get_id()] = stdout_thread
            stderr_threads[mcp.get_id()] = stderr_thread

            stdout_thread.start()
            stderr_thread.start()

        if block:
            try:
                self.termination_event.wait()
            except KeyboardInterrupt:
                print("Nanny interrupted")
            except Exception as e:
                print("Exception thrown:")
                print(e.args)
            finally:
                self.deconstruct()


            # time.sleep(time_out)
            # for id,cp in self.managed_child_processes.items():
            #     if cp.is_alive():
            #         print(("Child Process %s at %s did not terminate %s seconds after the first process in cluster terminated. Terminating now." %(cp.get_name(), cp.get_ip(), time_out)))
            #         cp.deconstruct()
            # for id,cp in self.managed_child_processes.items():
            #     if cp.is_alive():
            #         print(("Child Process %s at %s did not terminate. Force quitting now." %(cp.get_name(),cp.get_ip())))
            #         cp.deconstruct(signal.SIGKILL)



    def execute_all_child_processes_yield_results(self, time_out=5, result_time_out=None, stdout_stopping_criterium=lambda line:False, stderr_stopping_criterium =lambda line:False):
        '''
        Executes all child processes and starts managing communications. This method returns only when all child processes terminated.
        It might be the case that some child-processes hang or don't terminate. In this case, when one (the first) process
          terminates, all other processes are killed after time_out seconds. This behaviour is triggered also when

        :param: time_out: Grace Period for child processes to terminate before being killed
        :param: stdout_stopping_criterium: Receives the line from stdout. When evaluates to True, the stdout pipe is flushed and the termination request is set
        :param: stderr_stopping_criterium: Receives the line from stderr. When evaluates to True, the stdout pipe is flushed and the termination request is set
        :return:
        '''

        stdout_threads = {}
        stderr_threads = {}
        at_least_one_cp_started = False

        name_max_lenght = max([len(mcp.name) for mcp in self.managed_child_processes.values()])
        for i, id in enumerate(self.managed_child_processes.keys()):
            mcp =self.managed_child_processes[id]
            assert isinstance(mcp, RemotePythonProcess)
            if not mcp.is_generator:
                continue

            if mcp.monitor_for_termination:
                mcp.set_termination_event(self.termination_event)
            else:
                print("ALARM")

            stdin, stdout, stderr = mcp.execute_child_process()

            prefix = mcp.name.ljust(name_max_lenght)+": "

            # True if in debug mode
            gettrace = getattr(sys, 'gettrace', None)
            monitor_if_stuck_timeout = mcp.monitor_if_stuck_timeout if not gettrace() else None # only set timeout if not in debug mode

            #
            stdout_thread = threading.Thread(target=self._monitor_and_forward_child_communication,name="%s_stdout_Thread"%mcp.get_name(),
                                             args=(stdout,sys.stdout,mcp.name,mcp.get_termination_event(),stdout_stopping_criterium, prefix, monitor_if_stuck_timeout))
            stderr_thread = threading.Thread(target=self._monitor_and_forward_child_communication,name="%s_stderr_Thread"%mcp.get_name(),
                                             args=(stderr,sys.stderr,mcp.name,mcp.get_termination_event(),stderr_stopping_criterium, prefix, None))
            stdout_thread.setDaemon(True)
            stderr_thread.setDaemon(True)
            stdout_threads[mcp.get_id()] = stdout_thread
            stderr_threads[mcp.get_id()] = stderr_thread

            stdout_thread.start()
            stderr_thread.start()
            at_least_one_cp_started = True

        if not at_least_one_cp_started:
            ARTEMIS_LOGGER.warn("No RemotePythonProcess that is a generator has been started")
        multiplex_generator = self.multiplex_return_generators(result_time_out)
        try:
            for result in multiplex_generator:
                yield result
            time.sleep(time_out)
        except KeyboardInterrupt:
            self.deconstruct()
        finally:
            self.termination_event.set()

        for id, cp in self.managed_child_processes.items():
            if cp.is_alive():
                print(("Child Process %s at %s did not terminate %s seconds after the first process in cluster terminated. Terminating now." %(cp.get_name(), cp.get_ip(), time_out)))
                cp.kill(signal.SIGINT)

        for id,cp in self.managed_child_processes.items():
            if cp.is_alive():
                print(("Child Process %s at %s did not terminate. Force quitting now." %(cp.get_name(),cp.get_ip())))
                cp.deconstruct(signal.SIGKILL)

    def deconstruct(self):
        '''
        This method is called when SIGINT or SIGTERM are called.
        This aggressively deconstructs the Nanny and all child processes. Then, the signal is passed back to the original signal handlers.
        :return:
        '''
        for cp in self.managed_child_processes.values():
            if cp.is_alive():
                cp.kill(signal.SIGINT)
        time.sleep(1.0)
        for cp in self.managed_child_processes.values():
            if cp.is_alive():
                print(("Child Process %s at %s still alive, force terminating now"% (cp.name, cp.get_ip())))
                cp.kill(signum=signal.SIGKILL)
                cp.kill(signum=signal.SIGTERM)


    def _monitor_and_forward_child_communication(self, source_pipe, target_pipe,process_name, termination_request_event=None, stopping_criterium=None, prefix="", timeout=None):
        '''
        thread to forward communication from source_pipe to target_pipe
        :param source_pipe:ChildProcess
        :param target_pipe:
        :param termination_request_event: Is set once the source_pipe has closed and this thread terminates, or when ths stopping_criterium has been met.
        :param stopping_criterium:
        :param prefix:
        :return:
        '''
        if timeout is not None:
            line_printed_event = threading.Event()
            line_printed_event.clear()
            t = threading.Thread(target=self._output_monitoring_timer_thread,args=(process_name,line_printed_event,termination_request_event,timeout))
            t.setDaemon(True)
            t.start()
        with source_pipe:
            for line in iter(source_pipe.readline, b''):
                target_pipe.write("%s%s"%(prefix,line))
                target_pipe.flush()
                if timeout is not None:
                    line_printed_event.set()
                if stopping_criterium is not None and stopping_criterium(line):
                    print("Stopping criterium observed in process %s from Nanny %s" % ( process_name, self.name))
                    termination_request_event.set()
                    break
            if termination_request_event is not None:
                termination_request_event.set() # The input pipe closed, this thread terminates and we would like everybody to terminate
        # if timeout is not None:
        #     t.join()
        #     print("_output_monitoring_timer_thread has joined")

    def _output_monitoring_timer_thread(self, process_name, line_printed_event,termination_request_event, timeout=1800): # 5min
        timeout_wait_start = time.time()
        while time.time() - timeout_wait_start <= timeout:
            if line_printed_event.is_set():
                line_printed_event.clear()
                timeout_wait_start = time.time()
            if termination_request_event.is_set():
                return
            time.sleep(1.0)
        if termination_request_event.is_set():
            return

        print("Timeout occurred after %.1f min, process %s from Nanny %s stuck"%(timeout/60., process_name, self.name))
        termination_request_event.set()