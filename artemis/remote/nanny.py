import signal
import sys
import threading
import time

import os

from artemis.experiments.experiment_record import get_current_experiment_name, get_current_experiment_dir


class ManagedChildProcess(object):
    def __init__(self,cp,monitor_for_termination,monitor_if_stuck_timeout):
        self.cp = cp
        self.monitor_for_termination = monitor_for_termination
        self.monitor_if_stuck_timeout = monitor_if_stuck_timeout

    def get_process(self):
        return self.cp

    @property
    def name(self):
        return self.cp.name

    def monitor_if_stuck(self):
        return self.monitor_if_stuck_timeout is not None

    def kill(self, signal=signal.SIGINT):
        return self.cp.kill(signal)

    def is_alive(self):
        return self.cp.is_alive()

    def get_name(self):
        return self.cp.get_name()

    def get_ip(self):
        return self.cp.get_ip()

    def execute(self):
        return self.cp.execute_child_process()

    def get_id(self):
        return self.cp.get_id()

    def deconstruct(self,signum):
        return self.cp.deconstruct(signum)

class Nanny(object):
    '''
    Manages child processes. This class manages the start, live and deconstruction of child processes across different machines.
    '''
    def __init__(self):
        self.managed_child_processes = {}
        self.stdout_threads = {}
        self.original_sigint_handler = signal.getsignal(signal.SIGINT)
        self.original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self.deconstruct)
        signal.signal(signal.SIGTERM, self.deconstruct)
        # atexit.register(self.deconstruct)

    def register_child_process(self, cp, monitor_for_termination=True, monitor_if_stuck_timeout=None):
        '''
        This adds a child process to the Nanny.
        :param: monitor_for_termination: if set, this process' termination will trigger the shutdown process. If not set, this process' termination will go unnoticed
        :param: monitor_if_stuck_timeout: If set to a positive number, this process' communication is monitored. If not String has been passed over the Stdout channel of the child process in
         monitor_if_stuck_timeout seconds, the process is assumed to be stuck and the shutdown procedure is started.
        :return:
        '''
        assert monitor_if_stuck_timeout is None or (monitor_if_stuck_timeout > 0 and type(monitor_if_stuck_timeout) == int), "Please set monitor_if_stuck_timeout to a positive integer"

        self.managed_child_processes[cp.get_id()] = ManagedChildProcess(cp, monitor_for_termination,monitor_if_stuck_timeout)

    def get_child_processes(self):
        return {id:mcp.get_process() for id,mcp in self.managed_child_processes.iteritems()}

    def execute_all_child_processes(self, time_out=1, stdout_stopping_criterium=lambda line:False, stderr_stopping_criterium =lambda line:True):
        '''
        Executes all child processes and starts managing communications. This method returns only when all child processes terminated.
        It might be the case that some child-processes hang or don't terminate. In this case, when one (the first) process
          terminates, all other processes are killed after time_out seconds. This behaviour is triggered also when

        :param: time_out: Grace Period for child processes to terminate before being killed
        :param: stdout_stopping_criterium: Receives the line from stdout. When evaluates to True, the stdout pipe is flushed and the termination request is set
        :param: stderr_stopping_criterium: Receives the line from stderr. When evaluates to True, the stdout pipe is flushed and the termination request is set
        :return:
        '''

        termination_request_event = threading.Event()
        stdout_threads = {}
        stderr_threads = {}



        name_max_lenght = max([len(mcp.name) for mcp in self.managed_child_processes.values()])

        for i, id in enumerate(self.managed_child_processes.keys()):
            cp =self.managed_child_processes[id]
            stdin, stdout, stderr = cp.execute()

            prefix = cp.name.ljust(name_max_lenght)+": "

            # True if in debug mode
            gettrace = getattr(sys, 'gettrace', None)
            timeout = mcp.monitor_if_stuck_timeout if not gettrace() else None # only set timeout if not in debug mode


            stdout_thread = threading.Thread(target=self._monitor_and_forward_child_communication,
                                             args=(stdout,sys.stdout,cp.name,termination_request_event,stdout_stopping_criterium, prefix, timeout))
            stdout_thread.setDaemon(True)

            # stderr_stopping_criterium = lambda x: err_fun(cp.get_ip()) if terminate_at_error else None
            stderr_thread = threading.Thread(target=self._monitor_and_forward_child_communication,
                                             args=(stderr,sys.stderr,cp.name,termination_request_event,stderr_stopping_criterium, prefix, None))
            stderr_thread.setDaemon(True)
            stdout_threads[cp.get_id()] = stdout_thread
            stderr_threads[cp.get_id()] = stderr_thread

            stdout_thread.start()
            stderr_thread.start()

        try:
            while not termination_request_event.wait(0.01):
                pass
        except KeyboardInterrupt:
            print("Nanny interrupted")
            sys.exit(1)

        # Grace period for other threads to shutdown
        time.sleep(time_out)
        for id,cp in self.managed_child_processes.iteritems():
            if cp.is_alive():
                print("Child Process %s at %s did not terminate %s seconds after the first process in cluster terminated. Terminating now." %(cp.get_name(), cp.get_ip(), time_out))
                cp.kill()

        for id,cp in self.managed_child_processes.iteritems():
            if cp.is_alive():
                print("Child Process %s at %s did not terminate. Force quitting now." %(cp.get_name(),cp.get_ip()))
                cp.deconstruct(signal.SIGKILL)
        time.sleep(1.0)

        # This should return immediately, since the underlying pipes should have run dry. if it doesn't I messed up...:
        for stdout_thread, stderr_thread in zip(stdout_threads.values(), stderr_threads.values()):
            assert not stdout_thread.is_alive(), "This should not have happened"
            assert not stderr_thread.is_alive(), "This should not have happened"

    def deconstruct(self, signum, frame=None):
        '''
        This method is called when SIGINT or SIGTERM are called.
        This aggressively deconstructs the Nanny and all child processes. Then, the signal is passed back to the original signal handlers.
        :return:
        '''

        for cp in self.managed_child_processes.values():
            cp.kill()
        time.sleep(1.0)
        for cp in self.managed_child_processes.values():
            if cp.is_alive():
                print("Child Process %s at %s still alive, force terminating now"% (cp.name, cp.get_ip()))
                cp.kill(signal=signal.SIGTERM)

        signal.signal(signal.SIGINT, self.original_sigint_handler)
        signal.signal(signal.SIGTERM, self.original_sigterm_handler)
        os.kill(os.getpid(), signum)

    def _monitor_and_forward_child_communication(self, source_pipe, target_pipe,process_name, termination_request_event=None, stopping_criterium=None, prefix="", timeout=None):
        '''
        thread to forward communication from source_pipe to target_pipe
        :param source_pipe:
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
                if stopping_criterium is not None and stopping_criterium(line) and not "pydev debugger" in line and line.strip():
                    target_pipe.write(source_pipe.read())
                    target_pipe.flush()
                    break
            if termination_request_event is not None:
                termination_request_event.set() # The input pipe closed, this thread terminates and we would like everybody to terminate

    def _output_monitoring_timer_thread(self, process_name, line_printed_event,termination_request_event, timeout=1800): # 5min
        while not termination_request_event.wait(0.1):
            t_start = time.time()
            line_printed_event.wait(timeout)
            line_printed_event.clear()
            t_end = time.time()
            # print("Something was written, time since last line: %.3f"%(t_end-t_start))
            if t_end-t_start > timeout:
                if line_printed_event.is_set():
                    continue
                try:
                    exp_name = get_current_experiment_name()
                    curr_dir = get_current_experiment_dir()
                    with open(os.path.join(curr_dir,"experiment_stuck"),"wb"):
                        pass
                except:
                    exp_name=""
                print("Timeout occurred after %.1f min, process %s%s stuck"%(timeout/60., process_name, " from experiment %s"%exp_name if exp_name != "" else ""))
                termination_request_event.set()
                break