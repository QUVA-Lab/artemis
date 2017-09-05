import Queue
import os
import socket
import sys
import threading
import time
import uuid
import pickle
from collections import namedtuple
from artemis.general.should_be_builtins import is_lambda
from artemis.plotting.matplotlib_backend import get_plotting_server_address
from artemis.remote.child_processes import PythonChildProcess
from artemis.remote.nanny import Nanny
from artemis.remote.file_system import check_config_file
from artemis.remote.port_forwarding import forward_tunnel
from artemis.remote.utils import get_local_ips, send_size, recv_size, check_ssh_connection

_to_subprocess_queue = None
_id_queue = None
_nanny = None
comm_terminate_event = threading.Event()

DBPlotMessage = namedtuple('DBPlotMessage', ['plot_id', 'dbplot_args'])


def dbplot_remotely(arg_locals):
    """
    This method should be called from dbplot immedeatly, in case we should plot remotely.
    arg_locals con
    :param arg_locals: A dict of arguments with which dbplot was called.
    """
    global _to_subprocess_queue
    global _id_queue

    assert not is_lambda(arg_locals['plot_type']), "dbplot in server mode does not accept lambda. Use partial instead"

    if _to_subprocess_queue is None:
        # This is the first call to dbplot, we need to set up the server.
        set_up_plotting_server()

    # The data is being prepared and sent to the remote plotting server
    unique_plot_id = str(uuid.uuid4())
    data_to_send = DBPlotMessage(plot_id = unique_plot_id, dbplot_args=arg_locals)
    serialized_command = pickle.dumps(data_to_send, protocol=2)
    _to_subprocess_queue.put(serialized_command)

    # Now we wait (or not) for the remote plot to be rendered.
    # Todo: This does not work reliably at the moment when several concurrent threads use dbplot. One thread might have dequeued the plot_id that belongs the other thread, such that
    # the other thread will never receive the plot_id. In case the user specified a positive waiting duration, the second call will then not return earlier although the plot might
    # have been rendered already
    wait_for_display_sec = arg_locals["wait_for_display_sec"]
    if wait_for_display_sec != 0:
        if wait_for_display_sec <0:
            wait_for_display_sec = sys.maxint # Todo: There is no timeout here
        begin = time.time()
        time_left = wait_for_display_sec
        try:
            while _id_queue.get(timeout=time_left) != unique_plot_id:
                time_used = time.time() - begin
                time_left = wait_for_display_sec - time_used
                if time_left < 0:
                    break
        except Queue.Empty:
            pass

def deconstruct_plotting_server():
    global _nanny
    global _to_subprocess_queue
    global _id_queue

    if _nanny:
        _nanny.deconstruct()

    _to_subprocess_queue = None
    _id_queue = None
    _nanny = None

def set_up_plotting_server():
    """
    Sets up the plotting server.
    """

    print("Setting up Plotting Server")

    # First we generate the system call that starts the server
    # TODO: This assumes the same installation path relative to the home-dir on the local machine as on the remote machine
    file_to_execute = os.path.join(os.path.dirname(__file__), 'plotting_server.py')
    file_to_execute = file_to_execute.replace(os.path.expanduser("~"),"~",1)
    plotting_server_address = get_plotting_server_address()
    if plotting_server_address == "":
        plotting_server_address = "127.0.0.1"
    if plotting_server_address in get_local_ips():
        command = ["python", "-u", file_to_execute]
    else:
        check_config_file(plotting_server_address) # Make sure all things are set
        check_ssh_connection(plotting_server_address) # Make sure the SSH-connection works
        command =["export DISPLAY=:0.0;", "python","-u", file_to_execute]
        # TODO: Setting DISPLAY to :0.0 is a heuristic at the moment. I don't understand yet how these DISPLAY variables are set.

    # With the command set up, we can instantiate a child process and start it. Also we want to forward stdout and stderr from the remote process asynchronously.
    global _nanny
    _nanny = Nanny()
    cp = PythonChildProcess(ip_address=plotting_server_address, command=command, name="Plotting_Server",set_up_port_for_structured_back_communication=True)
    _nanny.register_child_process(cp,monitor_for_termination=False,monitor_if_stuck_timeout=None,)
    _nanny.execute_all_child_processes(blocking=False)
    back_comm_queue = cp.get_queue_from_cp()
    try:
        is_debug_mode = getattr(sys, 'gettrace', None)
        timeout = None if is_debug_mode() else 10
        server_message = back_comm_queue.get(block=True,timeout=timeout)
    except Queue.Empty:
        print("The Plotting Server did not respond for 10 seconds. It probably crashed")
        sys.exit(1)

    try:
        port = int(server_message.dbplot_message)
    except ValueError:
        print("There was an incorrect string on the remote server's stdout. Make sure the server first communicates a port number. Received:\n {}".format(server_message.dbplot_message))
        sys.exit(0)

    # In the remote setting we don't want to rely on the user correctly specifying their firewalls. Therefore we need to set up port forwarding through ssh:
    # Also, we have the ssh session open already, so why not reuse it.
    if plotting_server_address not in get_local_ips():
        ssh_conn = cp.get_ssh_connection()
        # Needs to be in a thread since the call blocks forever.
        # Todo: this ssh tunnel is opened system-wide. That means that any subsequent attempts to open the ssh-tunnel (by another dbplot-using process, for example)
        # will perform wiredly. As far as I have tested, nothing happenes and the port forwarfding works just fine in the second process, However when one of the two
        # processes terminates, the ssh-tunnel is closed for the other process as well.
        t3 = threading.Thread(target = forward_tunnel, kwargs={"local_port":port, "remote_host":plotting_server_address, "remote_port":port,"ssh_conn":ssh_conn})
        t3.setDaemon(True)
        t3.start()

    # Now attempt to connect to the plotting server
    server_address = ("localhost", port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(tuple(server_address))
    except:
        raise

    # Once connected, set up the asynchronous threads that forward every dbplot call to the server. We make this asynchronously for two reasons:
    # 1.) the queues involved can be shared between different threads (not processes), therefore allowing for asynchronous dbplot calls that are both correctly forwarded.
    # 2.) sending a plot away to the server now immediatly returns, independent on any socket-related communication delays that might exist.
    # (There shouldn't be any, but, you know, principle)
    global _to_subprocess_queue
    global _id_queue
    _to_subprocess_queue = Queue.Queue()
    _id_queue = Queue.Queue()
    t1 = threading.Thread(target=push_to_server, args=(_to_subprocess_queue, sock))
    t1.setDaemon(True)
    t1.start()
    # if blocking:
    t2 = threading.Thread(target=collect_from_server, args=(_id_queue, sock))
    t2.setDaemon(True)
    t2.start()


def push_to_server(queue, sock):
    while True:
        try:
            message = queue.get_nowait()
            send_size(sock, message)
        except Queue.Empty:
            time.sleep(0.01)
    sock.close()


def collect_from_server(queue, sock):
    while True:
        recv_message = recv_size(sock)
        queue.put(recv_message)
    sock.close()