from __future__ import print_function
# import matplotlib
# matplotlib.use('AGG')
import Queue
import argparse
import atexit
import sys
import os
import threading
import time
import pickle

import subprocess

from artemis.fileman.local_dir import get_relative_path, get_local_path
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from artemis.remote import set_forward_to_server
from artemis.remote.utils import get_socket, recv_size, send_size

sys.path.extend([os.path.expanduser('~/PycharmProjects/artemis')])

def send_port_if_running_and_join():
    port_file_path = get_local_path("tmp/plot_server/port.info", make_local_dir=True)
    if os.path.exists(port_file_path):
        with open(port_file_path, 'r') as f:
            port = pickle.load(f)
        print(port)
        print("Your dbplot call is attached to an existing plotting server. \nAll stdout and stderr of this existing plotting server "
              "is forwarded to the process that first created this plotting server. \nIn the future we might try to hijack this and provide you "
              "with these data streams")
        print("Use with care, this functionallity has some issues")
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
              "this message. Otherwise a previously run plotting server crashed without cleaning up afterwards. \nIn this case, please manually delete the file at {}".format(port_file_path))
    with open(port_file_path, 'wb') as f:
        pickle.dump(port,f)


def remove_port_file():
    print("Removing port file")
    port_file_path = get_local_path("tmp/plot_server/port.info", make_local_dir=True)
    os.remove(port_file_path)



def run_plotting_server(address, port):
    '''
    Address and port to listen on.
    :param address:
    :param port:
    :return:
    '''

    sock, port = get_socket(address=address,port=port)
    print(port)
    write_port_to_file(port)
    max_number_clients = 100
    max_plot_batch_size = 20000
    sock.listen(max_number_clients)
    print("Plotting Server is listening")

    main_input_queue = Queue.Queue()
    return_queue = Queue.Queue()
    t0 = threading.Thread(target=handle_socket_accepts,args=(sock, main_input_queue, return_queue, max_number_clients))
    t0.setDaemon(True)
    t0.start()

    #Now, we can accept plots in the main thread!
    while True:
        # First retrieve all data_points that might have come in in the mean-time:
        data_items = _queue_get_all(main_input_queue,max_plot_batch_size, block=False)
        if len(data_items) > 0:
            # print("Dequeued {} dbplot commands".format(len(data_items)))
            return_values = []
            with hold_dbplots():
                for data in data_items:
                    client = data["client"]
                    serialized_plot_message = data["plot"]
                    plot_message = pickle.loads(serialized_plot_message)
                    plot_id = plot_message["plot_id"]
                    plot_data = plot_message["data"]
                    plot_data["draw_now"] = False
                    dbplot(**plot_data)
                    return_values.append((client,plot_id))
            for client, plot_id in return_values:
                return_queue.put([client,plot_id])
                # return_queue.get()
        else:
            time.sleep(0.1)


def _queue_get_all(q, maxItemsToRetreive, block=False):
    items = []
    for numOfItemsRetrieved in range(0, maxItemsToRetreive):
        try:
            items.append(q.get(block=block))
        except Queue.Empty, e:
            break
    return items

def _queue_get_all_no_wait(q, maxItemsToRetreive):
    items = []
    for numOfItemsRetrieved in range(0, maxItemsToRetreive):
        try:
            items.append(q.get_nowait())
        except Queue.Empty, e:
            break
    return items


def handle_socket_accepts(sock, main_input_queue, return_queue, max_number):
    return_lock = threading.Lock()
    for _ in range(max_number):
        connection, client_address = sock.accept()
        # return_queue = Queue.Queue()
        # return_queues[client_address] = return_queue
        t0 = threading.Thread(target=handle_input_connection,args=(connection, client_address, main_input_queue))
        t0.setDaemon(True)
        t0.start()

        t1 = threading.Thread(target=handle_return_connection,args=(connection, client_address, return_queue, return_lock))
        t1.setDaemon(True)
        t1.start()




def handle_return_connection(connection, client_address, return_queue, return_lock):
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


def handle_input_connection(connection, client_address, input_queue):
    while True:
        recv_message = recv_size(connection)
        input_queue.put({"plot":recv_message,"client":client_address}, block=False)
    connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse', action="store_true", help='set if you want to merge plots on this server into one shared memory')
    args = parser.parse_args()
    if args.reuse is True:
        send_port_if_running_and_join()
    set_forward_to_server(False)
    run_plotting_server("0.0.0.0",7000)




