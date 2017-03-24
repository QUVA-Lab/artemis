from __future__ import print_function
from artemis.plotting.plotting_backend import set_server_plotting
from collections import namedtuple
import Queue
import argparse
import atexit
import sys
import os
import threading
import time
import pickle
from datetime import datetime
import signal
from artemis.fileman.local_dir import get_local_path, format_filename
from artemis.plotting.db_plotting import dbplot, hold_dbplots, set_dbplot_figure_size
from artemis.plotting.saving_plots import save_figure
from artemis.remote.utils import get_socket, recv_size, send_size
from matplotlib import pyplot as plt

sys.path.extend([os.path.dirname(os.path.dirname(__file__))])


def send_port_if_running_and_join():
    port_file_path = get_local_path("tmp/plot_server/port.info", make_local_dir=True)
    if os.path.exists(port_file_path):
        with open(port_file_path, 'r') as f:
            port = pickle.load(f)
        print(port)
        print("Your dbplot call is attached to an existing plotting server. \nAll stdout and stderr of this existing plotting server "
              "is forwarded to the process that first created this plotting server. \nIn the future we might try to hijack this and provide you "
              "with these data streams")
        print("Use with care, this functionallity might have unexpected side issues")
        try:
            while(True):
                time.sleep(20)
        except KeyboardInterrupt:
            print(" Redirected Server killed")
            sys.exit()
    else:
        with open(port_file_path,"w") as f:
            pass


def write_port_to_file(port):
    atexit.register(remove_port_file)
    port_file_path = get_local_path("tmp/plot_server/port.info", make_local_dir=True)
    if os.path.exists(port_file_path):
        print("port.info file already exists. This might either mean that you are running another plotting server in the background and want to start a second one.\nIn this case ignore "
              "this message. Otherwise a previously run plotting server crashed without cleaning up afterwards. \nIn this case, please manually delete the file at {}".format(port_file_path),
              file = sys.stderr)
    with open(port_file_path, 'wb') as f:
        pickle.dump(port,f)


def remove_port_file():
    print("Removing port file")
    port_file_path = get_local_path("tmp/plot_server/port.info", make_local_dir=True)
    if os.path.exists(port_file_path):
        os.remove(port_file_path)


def save_current_figure():
    print("Attempting to save figure")
    fig = plt.gcf()
    file_name = format_filename(file_string = '%T', current_time = datetime.now())
    save_path = get_local_path('output/{file_name}.pdf'.format(file_name=file_name))
    print("Current figure saved to {}".format(save_path))
    save_figure(fig,path=save_path)


class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("SIGINT caught")
        self.kill_now = True


def run_plotting_server(address, port):
    """
    Address and port to listen on.
    :param address:
    :param port:
    :return:
    """

    # Get the first available socket starting from portand communicate it with the client who started this server
    sock, port = get_socket(address=address, port=port)
    write_port_to_file(port)
    max_number_clients = 100
    max_plot_batch_size = 2000
    sock.listen(max_number_clients)
    print(port)
    print("Plotting Server is listening")

    # We want to save and rescue the current plot in case the plotting server receives a signal.SIGINT (2)
    killer = GracefulKiller()

    # The plotting server receives input in a queue and returns the plot_ids as a way of communicating that it has rendered the plot
    main_input_queue = Queue.Queue()
    return_queue = Queue.Queue()
    # Start accepting clients' communication requests
    t0 = threading.Thread(target=handle_socket_accepts,args=(sock, main_input_queue, return_queue, max_number_clients))
    t0.setDaemon(True)
    t0.start()

    # If killed, save the current figure
    atexit.register(save_current_figure)

    # Now, we can accept plots in the main thread!

    set_dbplot_figure_size(9,10)
    while True:
        if killer.kill_now:
            # The server has received a signal.SIGINT (2), so we stop receiving plots and terminate
            break
        # Retrieve data points that might have come in in the mean-time:
        client_messages = _queue_get_all_no_wait(main_input_queue,max_plot_batch_size)
        # client_messages is a list of ClientMessage objects
        if len(client_messages) > 0:
            return_values = []
            with hold_dbplots():
                for client_msg in client_messages:  # For each ClientMessage object
                    # Take apart the received message, plot, and return the plot_id to the client who sent it
                    plot_message = pickle.loads(client_msg.dbplot_message)  # A DBPlotMessage object (see plotting_client.py)
                    plot_message.dbplot_args['draw_now'] = False
                    axis = dbplot(**plot_message.dbplot_args)
                    axis.ticklabel_format(style='sci', useOffset=False)
                    return_values.append((client_msg.client_address, plot_message.plot_id))
                plt.rcParams.update({'axes.titlesize': 'small', 'axes.labelsize': 'small'})
                plt.subplots_adjust(hspace=0.4,wspace=0.6)
            for client, plot_id in return_values:
                return_queue.put([client,plot_id])
        else:
            time.sleep(0.1)


def _queue_get_all_no_wait(q, max_items_to_retreive):
    """
    Empties the queue, but takes maximally maxItemsToRetreive from the queue
    :param q:
    :param max_items_to_retreive:
    :return:
    """
    items = []
    for numOfItemsRetrieved in range(0, max_items_to_retreive):
        try:
            items.append(q.get_nowait())
        except Queue.Empty, e:
            break
    return items


def handle_socket_accepts(sock, main_input_queue, return_queue, max_number):
    """
    This Accepts max_number of incomming communication requests to sock and starts the threads that manages the data-transfer between the server and the clients
    :param sock:
    :param main_input_queue:
    :param return_queue:
    :param max_number:
    :return:
    """
    return_lock = threading.Lock()
    for _ in range(max_number):
        connection, client_address = sock.accept()
        t0 = threading.Thread(target=handle_input_connection,args=(connection, client_address, main_input_queue))
        t0.setDaemon(True)
        t0.start()

        t1 = threading.Thread(target=handle_return_connection,args=(connection, client_address, return_queue, return_lock))
        t1.setDaemon(True)
        t1.start()


def handle_return_connection(connection, client_address, return_queue, return_lock):
    """
    For each client, there is a thread that continously checks for the confirmation that a plot from this client has been rendered.
    This thread takes hold of the return queue, dequeues max 10 objects and checks if there is a return message for the client that is goverend by this thread.
    All other return messages are put back into the queue. Then the lock on the queue is released so that other threads might serve their clients their respecitve messages.
    The return messges belonging to this client are then sent back.
    :param connection:
    :param client_address:
    :param return_queue:
    :param return_lock:
    :return:
    """
    while True:
        return_lock.acquire()
        return_objects = _queue_get_all_no_wait(return_queue, 10)
        if len(return_objects) > 0:
            # print("Received {} return objects".format(len(return_objects)))
            owned_items = []
            for client, plot_id in return_objects:
                if client == client_address:
                    owned_items.append(plot_id)
                else:
                    return_queue.put((client,plot_id))
            return_lock.release()
            for plot_id in owned_items:
                message = plot_id
                send_size(sock=connection, data=message)
        else:
            return_lock.release()
            # print("no return value to send :)")
            time.sleep(0.01)


ClientMessage = namedtuple('ClientMessage', ['dbplot_message', 'client_address'])
# dbplot_args is a DBPlotMessage object
# client_address: A string IP address


def handle_input_connection(connection, client_address, input_queue):
    """
    For each client, there is a thread that waits for incoming plots over the network. If a plot came in, this plot is then put into the main queue from which the server takes
    plots away.
    :param connection:
    :param client_address:
    :param input_queue:
    :return:
    """
    while True:
        recv_message = recv_size(connection)
        input_queue.put(ClientMessage(recv_message, client_address))
        # input_queue.put({"plot":recv_message,"client":client_address}, block=False)
    connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse', action="store_true", help='set if you want to merge plots on this server into one shared memory')
    args = parser.parse_args()
    set_server_plotting(False)  # This causes dbplot to configure correctly
    if args.reuse is True:
        # TODO: has not been tested thoroughly yet. For example if you spawn two processes using the same server at the same time, there might be a conflict about who is the first who
        # actually sets up the server, and who joins the existing server
        send_port_if_running_and_join()

    run_plotting_server("0.0.0.0",7000) # We listen to the whole internet and start with port 7000
