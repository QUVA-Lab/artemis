from __future__ import print_function

import Queue
import argparse
import atexit
import pickle
import signal
import sys
import threading
import time
from datetime import datetime

import os
from matplotlib import pyplot as plt
import warnings
import matplotlib
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

#sys.path.extend([os.path.dirname(os.path.dirname(os.path.dirname(__file__)))])
from artemis.fileman.local_dir import get_artemis_data_path, format_filename
from artemis.plotting.matplotlib_backend import set_server_plotting
from artemis.plotting.db_plotting import dbplot, hold_dbplots, set_dbplot_figure_size
from artemis.plotting.saving_plots import save_figure
from artemis.remote.plotting.utils import _queue_get_all_no_wait, handle_socket_accepts
from artemis.remote.utils import get_socket, one_time_send_to


def send_port_if_running_and_join():
    port_file_path = get_artemis_data_path("tmp/plot_server/port.info", make_local_dir=True)
    if os.path.exists(port_file_path):
        with open(port_file_path, 'r') as f:
            port = pickle.load(f)
        print(port)
        print("Your dbplot call is attached to an existing plotting server. \nAll stdout and stderr of this existing plotting server "
              "is forwarded to the process that first created this plotting server. \nIn the future we might try to hijack this and provide you "
              "with these data streams")
        print("Use with care, this functionality might have unexpected side issues")
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
    port_file_path = get_artemis_data_path("tmp/plot_server/port.info", make_local_dir=True)
    if os.path.exists(port_file_path):
        # print("port.info file already exists. This might either mean that you are running another plotting server in the background and want to start a second one.\nIn this case ignore "
        #       "this message. Otherwise a previously run plotting server crashed without cleaning up afterwards. \nIn this case, please manually delete the file at {}".format(port_file_path),
        #       file = sys.stderr)
        # Keep for later development
        pass
    with open(port_file_path, 'wb') as f:
        pickle.dump(port,f)


def remove_port_file():
    port_file_path = get_artemis_data_path("tmp/plot_server/port.info", make_local_dir=True)
    if os.path.exists(port_file_path):
        os.remove(port_file_path)


def save_current_figure(path=""):
    fig = plt.gcf()
    file_name = format_filename(file_string = '%T', current_time = datetime.now())
    if path != "":
        save_path = os.path.join(path,"%s.pdf"%file_name)
    else:
        save_path = get_artemis_data_path('output/{file_name}.png'.format(file_name=file_name))
    save_figure(fig,path=save_path)


class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def run_plotting_server(address, port, client_address, client_port):
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
    one_time_send_to(address=client_address,port=client_port,message=str(port))

    # We want to save and rescue the current plot in case the plotting server receives a signal.SIGINT (2)
    killer = GracefulKiller()

    # The plotting server receives input in a queue and returns the plot_ids as a way of communicating that it has rendered the plot
    main_input_queue = Queue.Queue()
    return_queue = Queue.Queue()
    # Start accepting clients' communication requests
    t0 = threading.Thread(target=handle_socket_accepts,args=(sock, main_input_queue, return_queue, max_number_clients))
    t0.setDaemon(True)
    t0.start()


    # Received exp_dir on first db_plot_message?
    exp_dir_received = False
    # Now, we can accept plots in the main thread!

    set_dbplot_figure_size(9,10)
    while True:
        if killer.kill_now:
            sock.close() # will cause handle_socket_accepts thread to terminate
            # The server has received a signal.SIGINT (2), so we stop receiving plots and terminate
            break
        # Retrieve data points that might have come in in the mean-time:
        client_messages = _queue_get_all_no_wait(main_input_queue, max_plot_batch_size)
        # client_messages is a list of ClientMessage objects
        if len(client_messages) > 0:
            return_values = []
            with hold_dbplots():
                for client_msg in client_messages:  # For each ClientMessage object
                    # Take apart the received message, plot, and return the plot_id to the client who sent it
                    plot_message = pickle.loads(client_msg.dbplot_message)  # A DBPlotMessage object (see plotting_client.py)
                    plot_message.dbplot_args['draw_now'] = False

                    if not exp_dir_received:
                        if "exp_dir" == plot_message.dbplot_args["name"]:
                            atexit.register(save_current_figure,(plot_message.dbplot_args["data"]))
                            exp_dir_received = True
                            if len(client_messages) == 1:
                                continue
                            else:
                                continue
                    axis = dbplot(**plot_message.dbplot_args)
                    axis.ticklabel_format(style='sci', useOffset=False)
                    return_values.append((client_msg.client_address, plot_message.plot_id))
                if not exp_dir_received:
                    atexit.register(save_current_figure)
                    exp_dir_received = True
                plt.rcParams.update({'axes.titlesize': 'small', 'axes.labelsize': 'small'})
                plt.subplots_adjust(hspace=0.4,wspace=0.6)
            for client, plot_id in return_values:
                return_queue.put([client,plot_id])
        else:
            time.sleep(0.1)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse', action="store_true", help='set if you want to merge plots on this server into one shared memory')
    parser.add_argument('--address',type=str,default="",help='This is the address of the  plotting client that started the server.')
    parser.add_argument('--port',type=int,default=-1,help='This is the port on which the plotting client that started the server listens for the port of the plotting server.')
    args = parser.parse_args()
    set_server_plotting(False)  # This causes dbplot to configure correctly
    if args.reuse is True:
        # TODO: has not been tested thoroughly yet. For example if you spawn two processes using the same server at the same time, there might be a conflict about who is the first who
        # actually sets up the server, and who joins the existing server
        send_port_if_running_and_join()

    run_plotting_server("0.0.0.0",7100, args.address, args.port) # We listen to the whole internet and start with port 7000
