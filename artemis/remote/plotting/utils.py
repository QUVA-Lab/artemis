import socket
from six.moves import queue
import threading
import time
from collections import namedtuple
from artemis.remote.utils import recv_size, send_size


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
        except queue.Empty:
            break
    return items

def accept_connection_with_timeout_and_termination_event(sock,timeout=None,event=None):
    start = time.time()
    while timeout is None or (time.time() - start <= timeout):
        try:
            return sock.accept()
        except socket.timeout:
            if (event is not None and event.is_set()):
                break
    raise socket.timeout

def handle_socket_accepts(sock, main_input_queue=None, return_queue=None, max_number=1, name=None, timeout=None,received_termination_event=None):
    """
    This Accepts max_number of incoming communication requests to sock and starts the threads that manages the data-transfer between the server and the clients
    :param sock:
    :param main_input_queue:
    :param return_queue:
    :param max_number:
    :return:
    """
    return_lock = threading.Lock()
    connected_clients = 0
    threads = []
    sock.settimeout(0.1)
    sock.listen(1)
    while connected_clients < max_number:
        try:
            connection, client_address = accept_connection_with_timeout_and_termination_event(sock,timeout,received_termination_event)
        except socket.timeout:
            break
        else:
            connected_clients += 1
            if main_input_queue:
                t0 = threading.Thread(target=handle_input_connection, args=(connection, client_address, main_input_queue,received_termination_event),name=name)
                t0.setDaemon(True)
                t0.start()
                threads.append(t0)

            if return_queue:
                t1 = threading.Thread(target=handle_return_connection, args=(connection, client_address, return_queue, return_lock,received_termination_event),name=name)
                t1.setDaemon(True)
                t1.start()
                threads.append(t1)


def handle_return_connection(connection, client_address, return_queue, return_lock,received_termination_event=None):
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
        if not return_queue: break
        try:
            return_objects = _queue_get_all_no_wait(return_queue, 10)
        except Exception:
            break

        if len(return_objects) > 0:
            owned_items = []
            for client, plot_id in return_objects:
                if client == client_address:
                    owned_items.append(plot_id)
                else:
                    return_queue.put((client, plot_id))
            return_lock.release()
            for plot_id in owned_items:
                message = plot_id
                send_size(sock=connection, data=message)
        else:
            return_lock.release()
            time.sleep(0.01)


ClientMessage = namedtuple('ClientMessage', ['dbplot_message', 'client_address'])




def handle_input_connection(connection, client_address, input_queue,received_termination_event=None):
    """
    For each client, there is a thread that waits for incoming plots over the network. If a plot came in, this plot is then put into the main queue from which the server takes
    plots away.
    :param connection:
    :param client_address:
    :param input_queue:
    :return:
    """
    # if received_termination_event is not None:
    connection.settimeout(1.0) # for some reason, Mac needs this
    timeout = 1.0
    while True:
        if received_termination_event is not None and received_termination_event.is_set():
            # print("Termination Event was set. Stopping receiving from %s"%str(client_address))
            break
        else:
            try:
                recv_message = recv_size(connection,timeout=timeout)
            except socket.timeout:
                pass
            except Exception as e :
                raise
            else:
                if not input_queue: break
                input_queue.put(ClientMessage(recv_message, client_address))
    connection.close()
